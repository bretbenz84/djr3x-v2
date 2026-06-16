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
