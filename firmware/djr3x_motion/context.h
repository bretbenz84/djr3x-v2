// context.h — shared runtime state for the motion controller.
//
// One process-wide MotionContext holds everything the tasks touch. State is
// guarded by a cross-core spinlock (g_state_mux); all serial *writes* are
// serialized by g_tx_mux so NDJSON lines never interleave. Keep critical
// sections short: copy a snapshot under the lock, then format/emit outside it.
#pragma once
#include <Arduino.h>
#include "protocol.h"
#include "calib.h"   // boot defaults for the runtime-tunable drive params below

// ===== Small numeric helpers (shared) ======================================
inline float clampf(float v, float lo, float hi) {
  return v < lo ? lo : (v > hi ? hi : v);
}
inline uint32_t clampu(uint32_t v, uint32_t lo, uint32_t hi) {
  return v < lo ? lo : (v > hi ? hi : v);
}

// ===== Hard caps (compile-time ceilings; `config` can tighten, never exceed) =
// These are deliberately conservative for a tall, top-heavy droid on 2 driven
// wheels. Re-tune once the base is measured (docs/motion_system.md §14).
#define HARDCAP_MAX_LINEAR_MS     1.10f   // m/s — 0.60->0.80 when units became real (the base had
                                          // been field-driven ~0.72 m/s daily under the 4x cpm
                                          // miscalibration), then ->1.10 for the carpet surface
                                          // profile (full weight + pile drag need the headroom;
                                          // actual speed is drag-limited well below the cap)
#define HARDCAP_MAX_ANGULAR_RAD_S 6.10f   // rad/s (~350 deg/s) — with a 0.297 m track this
                                          // requests ~0.91 m/s per wheel in a pure pivot,
                                          // matching the 176 rpm / 100 mm drivetrain ceiling
#define HARDCAP_MAX_TURN_RATE_DPS 120.0f  // deg/s
#define HARDCAP_WATCHDOG_MS       2000u   // watchdog can't be set looser than this
#define HARDCAP_DRIVE_EXPIRY_MS   1000u
#define HARDCAP_WHEEL_TEST_MS     3000u   // single-wheel bring-up jog can't run longer

// ===== Runtime-tunable parameters (the `config` command, docs §10) ==========
struct MotionParams {
  float    max_lin       = 0.25f;  // m/s  ⚠ TEMP 2026-07-17: 60RPM left motor ceiling (was 0.35)
  float    max_ang       = 1.50f;  // rad/s (~86 deg/s) — turns felt slow at 1.05; tune with `set --max-ang`
  // FULL-SPEED collision envelope: the effective zones scale with measured speed,
  // from the STOP/SLOW_ZONE_MIN_M floors at rest (calib.h) up to these configured
  // values at ZONE_SPEED_REF_MS (stop_zone_eff/slow_zone_eff below). Retuned
  // 2026-07-11: fixed zones over-braked at range (the ±22.5° beams see off-path
  // clutter far out) and blocked precision parking near walls.
  float    slow_zone_m   = 0.75f;  // braking starts here at full speed (0.50->0.60 when units
                                   // became real: full teleop is now truly ~0.72 m/s)
  float    stop_zone_m   = 0.30f;  // hard-stop line at full speed (0.15->0.30 2026-07-16:
                                   // matrix ToF adds ~77 ms detection latency; at 0.6 m/s
                                   // the 0.15 envelope physically could not stop in time)

  float    come_stop_at_m= 0.60f;
  float    default_turn_deg  = 90.0f;
  float    default_turn_rate = 40.0f;   // deg/s
  uint32_t watchdog_ms       = 500;
  uint32_t drive_expiry_ms   = 300;
  uint32_t manual_idle_return_secs = 4;
  bool     manual_autoreturn = false;

  // Drive tuning (real HW only) — runtime-overridable via `config` for live
  // calibration + PID tuning without reflashing. Boot defaults from calib.h.
  float    kp = WHEEL_PID_KP;
  float    ki = WHEEL_PID_KI;
  float    kd = WHEEL_PID_KD;
  float    kff      = WHEEL_PID_KFF;    // velocity feedforward (duty per m/s of command)
  float    gain_scale_l = WHEEL_GAIN_SCALE_L;  // per-wheel multiplier on kp/ki/kff
  float    gain_scale_r = WHEEL_GAIN_SCALE_R;  // (mixed motors — see calib.h)
  float    min_duty = WHEEL_MIN_DUTY;   // running duty floor while rolling (duty)
  float    breakaway_duty = WHEEL_STRAIGHT_BREAKAWAY_DUTY;  // stall-gated dead-stop punch
                                        // for straight drive (pivots have their own tiers)
  float    accel_lin = DRIVE_ACCEL_LIN; // teleop linear setpoint slew (m/s^2)
  float    accel_ang = DRIVE_ACCEL_ANG; // teleop angular setpoint slew (rad/s^2)
  float    counts_per_meter_l = COUNTS_PER_METER_L;  // per-wheel: the 60RPM left
  float    counts_per_meter_r = COUNTS_PER_METER_R;  // motor has a different gearbox
  float    track_width_m    = TRACK_WIDTH_M;

  // Hallway steering assist (manual forward drive only — docs §6.3). While the pad
  // commands forward, ToF wall clearance steers the base away from walls / centers it
  // in a hallway; the operator's stick still adds on top and the stop reflex still
  // hard-blocks. All runtime-tunable via `config`.
  bool     assist_enabled   = true;
  float    assist_engage_mm = ASSIST_ENGAGE_MM;  // walls beyond this are ignored
  float    assist_gain      = ASSIST_GAIN;       // rad/s per METER of left-right imbalance
};

// ===== Speed-adaptive collision zones (calib.h ZONE_*/STOP_ZONE_MIN_M) ======
// Effective stop/slow distances as a function of MEASURED linear speed: the
// configured params are the full-speed envelope, shrinking linearly to the calib.h
// floors at rest. Used by BOTH the safety reflex (zone classification) and the
// control taper — keep them consuming these helpers so the two never disagree.
inline float zone_speed_frac(float lin_ms) {
  return clampf(fabsf(lin_ms) / ZONE_SPEED_REF_MS, 0.0f, 1.0f);
}
inline float stop_zone_eff(const MotionParams& p, float lin_ms) {
  const float lo = fminf(STOP_ZONE_MIN_M, p.stop_zone_m);   // a tighter config wins
  return lo + (p.stop_zone_m - lo) * zone_speed_frac(lin_ms);
}
inline float slow_zone_eff(const MotionParams& p, float lin_ms) {
  const float lo = fminf(SLOW_ZONE_MIN_M, p.slow_zone_m);
  const float s  = lo + (p.slow_zone_m - lo) * zone_speed_frac(lin_ms);
  return fmaxf(s, stop_zone_eff(p, lin_ms) + 0.05f);        // keep a real band above stop
}

// ===== Active finite command (turn / move / come) ===========================
struct FiniteCmd {
  CmdKind  kind = CMD_NONE;
  uint32_t seq  = 0;
  // targets / progress (units per docs §4)
  float    target_dist   = 0;   // m, signed (move)
  float    target_dtheta = 0;   // rad, signed magnitude consumed (turn)
  float    speed         = 0;   // m/s magnitude
  float    rate          = 0;   // rad/s magnitude
  float    progress_dist = 0;   // m, |displacement| accumulated
  float    progress_dtheta = 0; // rad, |rotation| accumulated
  // come bookkeeping
  bool     come_turning  = false;  // phase 1: rotating to heading
  float    come_stop_at  = 0;      // m
  float    come_sim_wall = 0;      // m, stub-only virtual wall ahead at start
  // wheel bring-up jog bookkeeping (CMD_WHEEL): raw single-wheel duty, time-bounded
  uint8_t  wheel_side    = 0;      // 0 = left, 1 = right
  float    wheel_frac    = 0;      // signed drive fraction -1..1 (+ = that wheel forward)
  uint32_t wheel_start_ms = 0;     // millis() at command start
  uint32_t wheel_ms      = 0;      // run duration (ms), then auto-stop -> done
};

// ===== Odometry / plant snapshot ===========================================
struct Odom {
  float x = 0, y = 0, theta = 0;  // m, m, rad (-pi,pi]
  float lin = 0, ang = 0;         // actual m/s, rad/s (plant output)
};

// ===== Setpoint the control loop drives toward =============================
struct Setpoint {
  float lin = 0;   // m/s target
  float ang = 0;   // rad/s target
  // Teleop spin↔arcade blend, 0..1 (gamepad only; autonomous paths leave it 1).
  //   0 = pure spin-in-place mixing (inside wheel may reverse)
  //   1 = full arcade clamp (inside wheel floored at 0, never reversed by a turn)
  // Computed in gamepad.cpp from the fwd-stick fraction (GAMEPAD_SPIN_BLEND_FWD_*)
  // and applied per-wheel in hal_drive_velocity, so the transition out of a spin is
  // a smooth tightening arc instead of a hard regime snap.
  float pivot_blend = 1.0f;
};

// ===== ToF distances (mm; -1 = sensor error) ==============================
// 8 radial sensors on the 540 mm base ring (docs/motion_protocol.md §6), mounted at
// the ring surface, every 45° starting 22.5° off the forward axis (no sensor on the
// cardinals themselves):
//   - 2 long-range  VL53L1X FRONT pair, ±22.5° off forward:  fl / fr
//   - 2 long-range  VL53L1X REAR  pair, ±22.5° off rearward: rl / rr
//   - 2 short-range VL53L0X LEFT  pair, ±22.5° off left:     lf / lb (front/back)
//   - 2 short-range VL53L0X RIGHT pair, ±22.5° off right:    rf / rb (front/back)
// The long pairs give room-scale wall sense fore/aft; the short pairs read the
// lateral clearance for the hallway steering assist. No down/cliff sensor. A reading
// is mm, -1 = sensor error, and a large value (per-type out-of-range cap) means
// "nothing in range = clear".
struct TofMm {
  int16_t fl = 4000, fr = 4000, rl = 4000, rr = 4000;  // long-range (VL53L1X) front/rear pairs
  int16_t lf = 1500, lb = 1500, rf = 1500, rb = 1500;  // short-range (VL53L0X) left/right pairs
};

// ===== Live gamepad mirror (telemetry only) ===============================
// A snapshot of the paired pad's stick + buttons THIS tick, for the GUI Motivator
// Control "physical controller" display. Written only by the MOTION_GAMEPAD_PRESENT
// build (gamepad.cpp); stays {connected=false} otherwise, so emit_telemetry reports
// "no pad". Normalized, deadzoned, GUI convention: lx right=+, ly stick-up=+.
struct GamepadLive {
  bool     connected = false;
  float    lx = 0.0f;        // turn axis  -1..1 (right = +)
  float    ly = 0.0f;        // drive axis -1..1 (stick-up = +)
  uint32_t btn_mask = 0;     // pressed buttons; bit order = GP_BTN_* in gamepad.cpp
  uint8_t  batt = 255;       // Bluepad32 Controller::battery() RAW 0..255 (semantics
                             // are fuzzy across the stack — forwarded raw, the Mac
                             // interprets; 255 = "not available" default per uni_controller.h)
};

// ===== IMU attitude (MPU-6050, telemetry + future fusion) ==================
// Complementary-filtered attitude from imu.cpp (50 Hz on the sensor task).
// pitch/roll are gravity-referenced (deg); yaw is bias-corrected gyro
// integration RELATIVE TO BOOT HEADING (deg, drifts slowly — no indoor
// magnetometer by design). ok=false when no MPU-6050 answered the boot probe.
// ===== Environment (BMP280/BME280 — env.cpp) ===============================
struct EnvState {
  bool  ok = false;       // sensor answered the boot probe and is still healthy
  float temp_c = 0.0f;    // air temperature, °C
  float hpa    = 0.0f;    // barometric pressure, hPa
  float rh     = -1.0f;   // relative humidity %, -1 = BMP280 fitted (no humidity)
};

// ===== Magnetometer (QMC5883L — mag.cpp; raw counts, host-side fusion) ======
struct MagState {
  bool    ok = false;     // sensor probed + healthy
  int16_t x = 0, y = 0, z = 0;   // RAW counts (±8 G range, 3000 LSB/gauss)
  bool    ovl = false;    // field overflow this sample — host rejects it
};

struct ImuState {
  bool  ok    = false;
  float pitch = 0.0f;   // deg, + = nose up
  float roll  = 0.0f;   // deg, + = right side down (chip frame; remap when mounted)
  float yaw   = 0.0f;   // deg, + = CCW from boot heading
};

// ===== Per-wheel drive diagnostics (telemetry only) =======================
// Measured wheel speed (m/s, encoder-derived) and the commanded PWM duty for each
// wheel THIS control tick — for diagnosing left/right asymmetry (a slower-ramping
// wheel, an encoder-scale mismatch, etc.). Written by the real HAL under the state
// lock; stays 0 in the stub build. Straight push: dl≈dr with vl<vr => the left
// drivetrain is physically weaker (PID hasn't compensated yet / soft tune);
// vl≈vr but the robot physically veers => an encoder is mis-scaled.
struct WheelDiag {
  float   vl = 0, vr = 0;     // measured wheel speed, m/s (encoder-derived)
  int16_t dl = 0, dr = 0;     // commanded duty, -PWM_DUTY_MAX..+PWM_DUTY_MAX
};

// ===== The whole shared state =============================================
struct MotionContext {
  MotionParams params;

  MotionState  state   = ST_IDLE;
  MotionOwner  owner   = OWNER_AUTO;
  MotionGamepad gamepad = GP_NONE;
  MotionFault  fault   = F_NONE;
  MotionZone   zone    = Z_CLEAR;
  MotionDir    blocked_dir = DIR_NONE;

  Setpoint  setpoint;     // commanded target (post-clamp, pre-safety)
  Odom      odom;         // integrated estimate / plant output
  TofMm     tof;          // latest sensor reads
  FiniteCmd finite;       // active finite command (kind==CMD_NONE if none)
  CmdKind   cmd_mode = CMD_NONE;  // what's currently driving motion

  // Manual (gamepad) override (docs §11). owner/gamepad above; these two below.
  bool      full_override = false;     // gamepad bypasses ToF zone/cliff gating (held)
  uint32_t  last_manual_input_ms = 0;  // last meaningful gamepad input (idle-autoreturn)
  // Pivot duty tiers currently in force (hal_drive_velocity reads them under the
  // state lock): breakaway while a wheel is STALLED in a pivot, run-floor while it
  // is ROLLING in a pivot (sustained sideways-scrub carry). Boot defaults from
  // calib.h; the gamepad's surface-mode toggle (L3) applies the hardwood/carpet
  // profile — a SURFACE property, so autonomous turns benefit from the mode too.
  float     spin_breakaway_duty = WHEEL_SPIN_BREAKAWAY_DUTY;
  float     spin_run_duty       = WHEEL_SPIN_RUN_DUTY;
  GamepadLive gp_live;                 // live pad mirror for the GUI (telemetry only)
  WheelDiag   wheels;                  // per-wheel measured speed + duty (telemetry diag)
  ImuState    imu;                     // MPU-6050 attitude (telemetry + future fusion)
  EnvState    env;                     // BMP280/BME280 room climate (telemetry)
  MagState    mag;                     // QMC5883L raw axes (host-side compass)
  bool        charging = false;        // on the charger (battery.cpp) — drive locked out

  uint32_t  cmd_seq   = 0;        // last applied command seq (telemetry)
  uint32_t  seq_alloc = 0;        // (unused on fw side; Mac allocates)
  uint32_t  errs      = 0;        // parse/framing error counter
  uint32_t  boot_id   = 0;        // random per boot

  // timers (millis timestamps)
  uint32_t  last_mac_ms   = 0;    // any valid Mac line resets this
  bool      seen_mac      = false;// watchdog arms only after first contact
  uint32_t  drive_set_ms  = 0;    // last `drive` receipt (deadman)

  int16_t   batt_mv = -1;         // pack voltage from the INA226; -1 = no sensor
                                  // (was a 12000 stub the host couldn't tell from
                                  // a real 12.0V pack)
  int16_t   batt_ma = 0;          // pack current, only when a real shunt is fitted
                                  // (BATT_SHUNT_MICROOHM > 0); + = discharging
  int8_t    batt_soc = -1;        // coulomb-counted state of charge, 0-100%;
                                  // -1 = unknown (no shunt / gauge not initialized)
};

// ===== Globals (defined in djr3x_motion.ino) ==============================
extern MotionContext     g_ctx;
extern SemaphoreHandle_t g_state_mux;  // recursive mutex guarding g_ctx
extern SemaphoreHandle_t g_tx_mux;     // serializes Serial writes

// Guards g_ctx. Recursive so an already-locked path can re-lock safely.
// RULE: never hold g_state_mux while taking g_tx_mux (emit_* take g_tx_mux) —
// always release state, THEN emit. This fixed lock order can't deadlock.
#define LOCK_STATE()   xSemaphoreTakeRecursive(g_state_mux, portMAX_DELAY)
#define UNLOCK_STATE() xSemaphoreGiveRecursive(g_state_mux)
