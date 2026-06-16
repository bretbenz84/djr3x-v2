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

// A wheel target below this (m/s magnitude) counts as "stopped" → the wheel is
// braked to zero and its integrator reset rather than chasing micro-setpoints.
#define WHEEL_STOP_EPS_MS  0.01f

// ---- ToF subsystem (5× VL53L0X) — only used when MOTION_TOF_PRESENT==1 ----
// Scaffold defaults; validate on hardware. The cliff floor/margin live in
// safety.cpp (the firmware only needs distances in mm here). docs §6.
#define TOF_COUNT             5
#define TOF_ADDR_BASE         0x30      // XSHUT sequencing assigns 0x30, 0x31, … in order
#define TOF_TIMEOUT_MS        50        // per-read I²C timeout
#define TOF_TIMING_BUDGET_US  33000     // 33 ms measurement budget (speed vs accuracy)
#define TOF_OUT_OF_RANGE_MM   8000      // VL53L0X returns ~8190 mm when nothing is in range
#define TOF_MUX_ADDR          0x70      // TCA9548A I²C address (only when MOTION_TOF_USE_MUX==1)
#define TOF_BOOT_SETTLE_MS    10        // settle time after raising each sensor's XSHUT

// ---- Bluetooth gamepad (Bluepad32) — only when MOTION_GAMEPAD_PRESENT==1 ----
// Left stick = arcade drive (Y forward, X turn); L1 creep / R1 boost; B = e-stop;
// Start = clear + return to AUTO; hold BOTH analog triggers = full-override (docs §11).
#define GAMEPAD_DEADZONE       0.12f    // stick fraction ignored around center
#define GAMEPAD_SCALE_CRUISE   0.65f    // default speed as a fraction of the caps
#define GAMEPAD_SCALE_CREEP    0.35f    // L1 held
#define GAMEPAD_SCALE_BOOST    1.00f    // R1 held
#define GAMEPAD_TRIGGER_MAX    1023.0f  // Bluepad32 analog trigger full-scale
#define GAMEPAD_FULL_OVERRIDE_FRAC 0.85f // both triggers past this fraction = bypass ToF
