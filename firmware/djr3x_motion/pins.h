// pins.h — ESP32-WROOM-32 GPIO assignments for the DJ-R3X drive base.
//
// Phase 1 hardware map (docs/motion_system.md §5,§6,§7). Only consulted when
// MOTION_HW_PRESENT==1; the stub build ignores all of this. Chosen to avoid the
// classic ESP32 footguns:
//   - GPIO 1/3   are the USB-serial link to the Mac (the whole protocol). Reserved.
//   - GPIO 6..11 are wired to the SPI flash. Never usable.
//   - GPIO 34..39 are input-only with NO internal pull-ups (used only for the
//     Hall inputs, which are push-pull — see below — so no pull-up is needed).
//   - GPIO 0/2/12/15 are strapping pins; 12 in particular bricks boot if pulled
//     high at reset. Kept off the motor/encoder lines.
//
// If your wiring differs, THIS is the one file to edit — nothing else hard-codes
// a pin number.
#pragma once

// ---- Drive motors: 2× BTS7960 full H-bridge, one per wheel ----------------
// Drive a wheel by PWM-ing exactly one of RPWM/LPWM (the other at 0); never both.
// R_EN+L_EN of each driver are tied together to a single enable GPIO (pull LOW to
// coast/disable). hal.cpp drives these via LEDC PWM (the channel mapping is handled
// there for both Arduino-ESP32 core 2.x and 3.x).
#define PIN_L_RPWM   16   // left  motor, forward duty (RPWM)
#define PIN_L_LPWM   17   // left  motor, reverse duty (LPWM)
#define PIN_R_RPWM   18   // right motor, forward duty (RPWM)
#define PIN_R_LPWM   19   // right motor, reverse duty (LPWM)
#define PIN_L_EN     23   // left  driver enable (R_EN+L_EN tied)
#define PIN_R_EN     27   // right driver enable (R_EN+L_EN tied)

// ---- Hall quadrature encoders (one A/B pair per wheel) --------------------
// JGB37-520 Hall outputs are push-pull at the encoder's VCC. POWER THE ENCODER
// AT 3.3 V so A/B swing 0–3.3 V (5 V would over-volt these GPIOs). Common ground
// with the ESP32 and the 12 V motor-supply ground is mandatory.
// Decoded by the PCNT peripheral (ESP32Encoder, full-quad x4). If a wheel counts
// the "wrong" way when driven forward, flip its sign in calib.h (ENC_SIGN_*) or
// swap the A/B pins here — REP-103 wants +counts = forward.
#define PIN_ENC_L_A  32   // left  encoder channel A (C1)
#define PIN_ENC_L_B  33   // left  encoder channel B (C2)
#define PIN_ENC_R_A  25   // right encoder channel A (C1)
#define PIN_ENC_R_B  26   // right encoder channel B (C2)

// ---- I2C — the ToF bus: 8 sensors behind a TCA9548A mux (docs §6) ----------
// Only consulted when MOTION_TOF_PRESENT==1 (tof.cpp). The 8 sensors (4× VL53L0X on
// mux ch 0-3 + 4× VL53L1X on mux ch 4-7) all share this one bus; the mux selects one
// at a time, so NO XSHUT GPIOs are needed (4/5/13/14/15 are free). XSHUT sequencing is
// unsupported for this layout (8 sensors > free GPIOs — tof.cpp #errors on it), so
// there are no PIN_TOF_XSHUT_* defines.
#define PIN_I2C_SDA  21
#define PIN_I2C_SCL  22
