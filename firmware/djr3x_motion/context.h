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
#define HARDCAP_MAX_LINEAR_MS     0.40f   // m/s
#define HARDCAP_MAX_ANGULAR_RAD_S 2.50f   // rad/s (~143 deg/s)
#define HARDCAP_MAX_TURN_RATE_DPS 120.0f  // deg/s
#define HARDCAP_WATCHDOG_MS       2000u   // watchdog can't be set looser than this
#define HARDCAP_DRIVE_EXPIRY_MS   1000u

// ===== Runtime-tunable parameters (the `config` command, docs §10) ==========
struct MotionParams {
  float    max_lin       = 0.25f;  // m/s
  float    max_ang       = 1.05f;  // rad/s (~60 deg/s)
  float    slow_zone_m   = 0.60f;
  float    stop_zone_m   = 0.25f;
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
  float    counts_per_meter = COUNTS_PER_METER;
  float    track_width_m    = TRACK_WIDTH_M;
};

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
};

// ===== ToF distances (mm; -1 = sensor error) ==============================
// 8 radial sensors for spatial awareness (docs/motion_protocol.md §6):
//   - 4 long-range VL53L1X at the CARDINALS: front / rear / left / right
//   - 4 short-range VL53L0X at the 45° DIAGONALS: fl / fr / rl / rr
// No down/cliff sensor in this layout. A reading is mm, -1 = sensor error, and a
// large value (per-type out-of-range cap) means "nothing in range = clear".
struct TofMm {
  int16_t front = 2000, rear = 2000, left = 2000, right = 2000;  // long-range (VL53L1X)
  int16_t fl = 1500, fr = 1500, rl = 1500, rr = 1500;            // short-range (VL53L0X)
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
  GamepadLive gp_live;                 // live pad mirror for the GUI (telemetry only)

  uint32_t  cmd_seq   = 0;        // last applied command seq (telemetry)
  uint32_t  seq_alloc = 0;        // (unused on fw side; Mac allocates)
  uint32_t  errs      = 0;        // parse/framing error counter
  uint32_t  boot_id   = 0;        // random per boot

  // timers (millis timestamps)
  uint32_t  last_mac_ms   = 0;    // any valid Mac line resets this
  bool      seen_mac      = false;// watchdog arms only after first contact
  uint32_t  drive_set_ms  = 0;    // last `drive` receipt (deadman)

  int16_t   batt_mv = 12000;      // stubbed pack voltage
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
