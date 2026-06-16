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

void hal_init();
void hal_read_tof(TofMm& out);                   // latest ToF distances (mm; -1 = error)

#if MOTION_HW_PRESENT
// Real closed-loop drive. These run inside control_tick UNDER the state lock, so
// they must be fast and non-blocking and must never take the serial mux.
void hal_read_odom(Odom& out, float dt);              // encoders -> odom (+ measured wheel speeds)
void hal_drive_velocity(float lin, float ang, float dt); // body vel -> kinematics -> per-wheel PID -> PWM
void hal_motors_off();                                // disable the H-bridges + reset PID (estop/idle)
#else
void hal_apply_velocity(float lin, float ang);        // stub no-op (commanded body velocity)
#endif
