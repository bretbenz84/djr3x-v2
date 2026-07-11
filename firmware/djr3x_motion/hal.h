// hal.h — hardware abstraction layer.
//
// MOTION_HW_PRESENT gates the real peripheral drivers. While it is 0 (nothing
// wired) the motor output is a no-op and the ToF read returns "all clear", so the
// whole protocol/control/safety stack runs and is testable on a bare ESP32. The
// default below stays 0 so the repo keeps building bare-board (and the smoke test
// keeps passing); the live device is built with -DMOTION_HW_PRESENT=1 (see
// README "Build modes"). Real drivers live in hal.cpp / pins.h / calib.h
// (docs/motion_system.md §5 motors, §6 ToF, §7 control loop).
#pragma once
#include "context.h"

#ifndef MOTION_HW_PRESENT
#define MOTION_HW_PRESENT 0    // override per-build: -DMOTION_HW_PRESENT=1
#endif

// MOTION_TOF_PRESENT gates the 8-sensor ToF subsystem (long VL53L1X front/rear pairs
// at ±22.5° off each axis + short VL53L0X left/right pairs) SEPARATELY from the drive motors,
// because the base can have working motors/encoders while the ToF sensors are still
// unwired. While it is 0 the ToF read returns "all clear" (obstacle avoidance
// inactive); the live drive build (MOTION_HW_PRESENT=1) ships this way today. Build
// with -DMOTION_TOF_PRESENT=1 once the sensors are wired. The real driver lives in
// tof.cpp (docs/motion_system.md §6).
#ifndef MOTION_TOF_PRESENT
#define MOTION_TOF_PRESENT 0   // override per-build: -DMOTION_TOF_PRESENT=1
#endif
// Addressing scheme for the 8 sensors on one I²C bus (docs §6.1). 1 = TCA9548A I²C
// multiplexer (all sensors stay at 0x29; the mux selects one channel at a time — uses
// NO extra GPIOs). The mux is REQUIRED for this layout: 8 sensors exceed the ESP32's
// free GPIOs for XSHUT sequencing (the -DMOTION_TOF_USE_MUX=0 path #errors in tof.cpp).
#ifndef MOTION_TOF_USE_MUX
#define MOTION_TOF_USE_MUX 1   // override per-build: -DMOTION_TOF_USE_MUX=0 (unsupported: 8>GPIOs)
#endif

void hal_init();
void hal_tof_init();                             // bring up the ToF subsystem (no-op in the stub)
void hal_read_tof(TofMm& out);                   // latest ToF distances (mm; -1 = error)

#if MOTION_HW_PRESENT
// Real closed-loop drive. These run inside control_tick UNDER the state lock, so
// they must be fast and non-blocking and must never take the serial mux.
void hal_read_odom(Odom& out, float dt);              // encoders -> odom (+ measured wheel speeds)
void hal_drive_velocity(float lin, float ang, float dt, bool pivot_steer, float pivot_blend); // body vel -> per-wheel PID -> PWM (pivot_steer: joystick; pivot_blend 0 spin .. 1 arcade morphs the mixing smoothly)
void hal_drive_wheel_raw(int side, float frac);       // ONE wheel at raw duty (side 0=L/1=R, frac -1..1) — bring-up jog, NO kinematics/PID
void hal_motors_off();                                // disable the H-bridges + reset PID (estop/idle)
#else
void hal_apply_velocity(float lin, float ang);        // stub no-op (commanded body velocity)
#endif
