// calib.h — measured drive-base constants + control gains (Phase 1).
//
// ⚠ THE GEOMETRY VALUES BELOW ARE PLACEHOLDERS, NOT MEASUREMENTS. ⚠
// They make odometry self-consistent but NOT accurate. Distances/angles will be
// wrong until you measure them on the real base (docs/motion_system.md §14):
//   1. Spin one wheel output-rev by hand; confirm COUNTS_PER_REV_OUTPUT.
//   2. Drive a measured 1.0 m straight; scale COUNTS_PER_METER by the error.
//   3. Spin a measured 360°; scale TRACK_WIDTH_M by the error.
// None of these affect safety (motion still needs an explicit command and the
// caps/watchdog/estop are independent) — only how far/how accurately it moves.
//
// The geometry values + the PID gains below are the BOOT DEFAULTS: they seed
// MotionParams and are runtime-overridable via the `config` command
// (counts_per_meter, track_width_m, kp/ki/kd), so you can calibrate + tune live with
// firmware/tools/motion_bench.py WITHOUT reflashing each iteration. Bake the winning
// values back here (or push them from config.py / .env) once you're happy.
#pragma once
#include <math.h>

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

// ---- Drive geometry (MEASURE — see above) --------------------------------
// Encoder: 11 cycles/motor-rev/channel × 176:1 gear × 4 (full quadrature)
//   ≈ 7744 counts per WHEEL output-rev. Verify empirically.
#define COUNTS_PER_REV_OUTPUT  7744.0f
#define WHEEL_DIAMETER_M       0.080f    // ⚠ placeholder — measure your wheel
#define TRACK_WIDTH_M          0.200f    // ⚠ placeholder — distance between drive wheels

// Counts per metre of wheel travel = counts/rev ÷ wheel circumference.
#define WHEEL_CIRCUM_M   ((float)M_PI * WHEEL_DIAMETER_M)
#define COUNTS_PER_METER (COUNTS_PER_REV_OUTPUT / WHEEL_CIRCUM_M)

// Per-wheel count direction. +1 means "driving the wheel forward makes its count
// increase." Flip to -1 (per wheel) if the bench hand-turn test shows the sign
// backwards, instead of rewiring A/B.
#define ENC_SIGN_L  (+1.0f)
#define ENC_SIGN_R  (+1.0f)

// Per-wheel MOTOR direction. +1 means "a positive (forward) duty spins the wheel
// forward." Flip to -1 (per wheel) — the software equivalent of swapping that motor's
// two power leads — if the bench `spin` test runs away / auto-estops or the wheel spins
// BACKWARD on a forward command. MUST agree with ENC_SIGN_* (forward duty -> forward
// travel -> +counts): a mismatch makes the velocity PID positive feedback and trips the
// runaway guard. Fixing direction HERE keeps each channel paired with its own encoder,
// so it does NOT desync odometry the way swapping only the motor leads does.
#define MOTOR_SIGN_L  (+1)
#define MOTOR_SIGN_R  (+1)

// ---- Motor PWM (LEDC) -----------------------------------------------------
// 20 kHz is above audible and well within the BTS7960's switching range.
#define PWM_FREQ_HZ    20000
#define PWM_RES_BITS   10                 // 0..1023 duty
#define PWM_DUTY_MAX   ((1 << PWM_RES_BITS) - 1)

// ---- Per-wheel velocity PID (target m/s → duty) ---------------------------
// ⚠ Starting gains — tune on the bench (docs §14.3): raise KP until the wheel
// tracks a step without buzzing, add KI to kill steady-state error, keep KD ~0
// unless it's oscillating. Units: duty per (m/s) of error.
#define WHEEL_PID_KP     1800.0f
#define WHEEL_PID_KI      900.0f
#define WHEEL_PID_KD        0.0f
// Anti-windup: the integral's duty contribution is clamped to this so it can't
// accumulate while the motor is saturated/stalled.
#define WHEEL_PID_I_CLAMP  (0.8f * PWM_DUTY_MAX)

// ---- Velocity feedforward + stiction compensation -------------------------
// The PID alone starts every move from ZERO duty and only reaches a useful duty
// once the integrator winds up — so low speeds sit below breakaway friction (weak
// + slow to start) and duty scales with speed (strong only when fast). Two terms
// fix that mechanically:
//   KFF  — feedforward duty per commanded m/s. The instant a speed is commanded the
//          wheel gets ~the right duty, so the loop only trims instead of building
//          from nothing. Start ≈ 0.9 * PWM_DUTY_MAX / max_lin (so the top commanded
//          speed maps to ~90% duty, leaving PID headroom). Tune on the bench.
//   MIN_DUTY — a fixed breakaway "kick" added in the travel direction whenever a
//          nonzero speed is commanded, to clear static friction on a heavy base.
//          Raise until the wheel starts moving crisply at creep; lower if it lurches.
#define WHEEL_PID_KFF    2600.0f    // duty per (m/s) of COMMAND (feedforward)
#define WHEEL_MIN_DUTY    120.0f    // stiction breakaway kick (duty), in travel dir

// A wheel target below this (m/s magnitude) counts as "stopped" → the wheel is
// braked to zero and its integrator reset rather than chasing micro-setpoints.
#define WHEEL_STOP_EPS_MS  0.01f

// ---- Drive setpoint slew (teleop feel) ------------------------------------
// Acceleration limit applied to the TELEOP (gamepad drive) setpoint so the base
// ramps smoothly toward the stick command in BOTH directions — symmetric, so a
// released stick coasts to a stop over ~(speed/accel) seconds instead of slamming
// to zero and dynamic-braking (the abrupt-stop complaint). Autonomous finite
// move/turn/come commands are NOT slewed here (they stay crisp + distance-accurate).
// Softened twice after field tests ("takes off super quick" x2): at max_lin≈0.35,
// 0.3 m/s² ramps 0→top in ~1.17 s. Tune with `set --accel-lin` (higher = snappier).
#define DRIVE_ACCEL_LIN    0.3f     // m/s^2  (teleop linear setpoint slew)
#define DRIVE_ACCEL_ANG    4.0f     // rad/s^2 (teleop angular setpoint slew)

// ---- ToF subsystem (8 radial sensors) — only used when MOTION_TOF_PRESENT==1 -
// 4× short-range VL53L0X on mux ch 0-3 (45° diagonals) + 4× long-range VL53L1X on
// mux ch 4-7 (cardinals). Requires the TCA9548A mux (8 sensors > free XSHUT GPIOs).
// Scaffold defaults; validate on hardware. docs §6.
#define TOF_SHORT_COUNT       4         // VL53L0X (short), mux ch 0..3 — 45° diagonals
#define TOF_LONG_COUNT        4         // VL53L1X (long),  mux ch 4..7 — cardinals
#define TOF_COUNT             (TOF_SHORT_COUNT + TOF_LONG_COUNT)   // 8 total
#define TOF_MUX_ADDR          0x70      // TCA9548A I²C address (mux selects one ch at a time)
// Per-read wait for a fresh continuous sample. MUST exceed the slowest sensor's
// inter-measurement period (L1X, below) or a live-but-slow sensor reads as -1 while
// we wait for its next sample. Dead sensors are skipped (s_ok), so this only bounds
// a genuinely stuck live sensor.
#define TOF_TIMEOUT_MS        100

// VL53L0X (short range, ~1.2 m reliable):
#define TOF_L0X_TIMING_BUDGET_US  33000 // 33 ms measurement budget (speed vs accuracy)
#define TOF_L0X_OUT_OF_RANGE_MM   2000  // clamp "nothing in range" to a far/clear value

// VL53L1X (long range): Long mode reaches ~4 m; needs a larger timing budget than L0X.
// The inter-measurement period must be >= the timing budget (+overhead), else the
// sensor won't produce readings (datasheet); 60 ms > 50 ms budget satisfies that.
#define TOF_L1X_TIMING_BUDGET_US  50000 // 50 ms budget (Long mode wants >= ~33 ms)
#define TOF_L1X_INTERMEASUREMENT_MS 60  // continuous-mode period (> timing budget)
#define TOF_L1X_OUT_OF_RANGE_MM   4000  // clamp "nothing in range" to a far/clear value

// ---- Bluetooth gamepad (Bluepad32) — only when MOTION_GAMEPAD_PRESENT==1 ----
// Left stick = arcade drive (Y forward, X turn); L1 creep / R1 boost; B = e-stop;
// Start = clear + return to AUTO; hold BOTH analog triggers = full-override (docs §11).
#define GAMEPAD_DEADZONE       0.12f    // stick fraction ignored around center
#define GAMEPAD_SCALE_CRUISE   0.85f    // default speed as a fraction of the caps
#define GAMEPAD_SCALE_CREEP    0.35f    // L1 held
#define GAMEPAD_SCALE_BOOST    1.00f    // R1 held
#define GAMEPAD_TRIGGER_MAX    1023.0f  // Bluepad32 analog trigger full-scale
#define GAMEPAD_FULL_OVERRIDE_FRAC 0.85f // both triggers past this fraction = bypass ToF
#define GAMEPAD_TRIGGER_PRESS_FRAC 0.50f // trigger past this fraction = "pressed" (GUI mirror)
