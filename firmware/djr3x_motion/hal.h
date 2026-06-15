// hal.h — hardware abstraction layer.
//
// MOTION_HW_PRESENT gates real peripheral drivers. While it is 0 (nothing wired)
// the motor output is a no-op and the ToF read returns "all clear", so the whole
// protocol/control/safety stack runs and is testable on a bare ESP32. Flip to 1
// and fill the marked sections as the BTS7960 drivers, Hall encoders, and
// VL53L0X sensors are wired (docs/motion_system.md §5,§6).
#pragma once
#include "context.h"

#define MOTION_HW_PRESENT 0    // 0 = stubbed bring-up build; 1 = real peripherals

void hal_init();
void hal_apply_velocity(float lin, float ang);   // commanded body velocity -> motors
void hal_read_tof(TofMm& out);                   // latest ToF distances (mm; -1 = error)
