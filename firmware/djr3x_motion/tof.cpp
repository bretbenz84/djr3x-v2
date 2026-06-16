// tof.cpp — ToF (Time-of-Flight) distance subsystem: 5× VL53L0X.
//
// Owns hal_read_tof() / hal_tof_init() for ALL builds, gated by MOTION_TOF_PRESENT
// (hal.h) independently of the motor drivers — the base can drive on real motors +
// encoders while the ToF sensors are still unwired. Until they are wired, the stub
// reports a clear room (obstacle avoidance inactive); safety.cpp's zone/cliff reflex
// runs against whatever this returns.
//
// ⚠ SCAFFOLD — the real-sensor paths below are NOT yet hardware-validated. The
//   addressing sequence, sensor→field mapping, timing budget, and the down-sensor
//   cliff calibration (CLIFF_FLOOR_MM/CLIFF_MARGIN_MM in safety.cpp) all need
//   bench checking once the sensors are physically on the bus. See docs §6, §14.
//
// Two addressing schemes share one I²C bus (docs §6.1), picked at build time:
//   MOTION_TOF_USE_MUX==0  XSHUT sequencing — every sensor powers up at 0x29, so we
//                          hold all in reset, then bring each up alone and reassign it
//                          0x30, 0x31, … (one XSHUT GPIO per sensor, pins.h).
//   MOTION_TOF_USE_MUX==1  TCA9548A mux — all sensors keep 0x29; select one channel at
//                          a time (no XSHUT GPIOs). Uses the Pololu VL53L0X lib.
#include "hal.h"            // MOTION_TOF_PRESENT / MOTION_TOF_USE_MUX + TofMm (via context.h)

#if MOTION_TOF_PRESENT
// ===========================================================================
// REAL SENSORS — VL53L0X ×5 (Pololu vl53l0x-arduino: supports setAddress()).
// ===========================================================================
#include "pins.h"
#include "calib.h"
#include <Arduino.h>
#include <Wire.h>
#include <VL53L0X.h>

static VL53L0X s_tof[TOF_COUNT];   // index order == placement order == TofMm fields

static inline int read_mm(int i);  // forward decl (defined per addressing scheme)

#if MOTION_TOF_USE_MUX
// ---- TCA9548A multiplexer ----
static const uint8_t TOF_MUX_CH[TOF_COUNT] = {0, 1, 2, 3, 4};  // sensor i -> mux channel

static void mux_select(uint8_t ch) {
  Wire.beginTransmission(TOF_MUX_ADDR);
  Wire.write(1 << ch);
  Wire.endTransmission();
}

void hal_tof_init() {
  Wire.begin(PIN_I2C_SDA, PIN_I2C_SCL);
  for (int i = 0; i < TOF_COUNT; i++) {
    mux_select(TOF_MUX_CH[i]);                 // only this sensor is on the bus now
    s_tof[i].setTimeout(TOF_TIMEOUT_MS);
    s_tof[i].init();                           // each keeps the default 0x29 behind the mux
    s_tof[i].setMeasurementTimingBudget(TOF_TIMING_BUDGET_US);
    s_tof[i].startContinuous();
  }
}

static inline int read_mm(int i) {
  mux_select(TOF_MUX_CH[i]);
  const int mm = s_tof[i].readRangeContinuousMillimeters();
  if (s_tof[i].timeoutOccurred() || mm >= TOF_OUT_OF_RANGE_MM) return -1;
  return mm;
}

#else
// ---- XSHUT sequencing (one GPIO per sensor) ----
static const int XSHUT[TOF_COUNT] = {
  PIN_TOF_XSHUT_FL, PIN_TOF_XSHUT_FC, PIN_TOF_XSHUT_FR, PIN_TOF_XSHUT_REAR, PIN_TOF_XSHUT_DOWN,
};

void hal_tof_init() {
  Wire.begin(PIN_I2C_SDA, PIN_I2C_SCL);
  // Hold every sensor in reset so the bus starts with nothing at 0x29.
  for (int i = 0; i < TOF_COUNT; i++) {
    pinMode(XSHUT[i], OUTPUT);
    digitalWrite(XSHUT[i], LOW);
  }
  delay(TOF_BOOT_SETTLE_MS);
  // Bring up one at a time; while a sensor is the only one at 0x29, move it to a
  // unique address so the rest can join without colliding. (Pololu multi-sensor
  // pattern: setAddress() writes via the current 0x29, then init() uses the new one.)
  for (int i = 0; i < TOF_COUNT; i++) {
    digitalWrite(XSHUT[i], HIGH);
    delay(TOF_BOOT_SETTLE_MS);
    s_tof[i].setTimeout(TOF_TIMEOUT_MS);
    s_tof[i].setAddress(TOF_ADDR_BASE + i);
    s_tof[i].init();
    s_tof[i].setMeasurementTimingBudget(TOF_TIMING_BUDGET_US);
    s_tof[i].startContinuous();
  }
}

static inline int read_mm(int i) {
  const int mm = s_tof[i].readRangeContinuousMillimeters();
  if (s_tof[i].timeoutOccurred() || mm >= TOF_OUT_OF_RANGE_MM) return -1;
  return mm;
}
#endif  // MOTION_TOF_USE_MUX

void hal_read_tof(TofMm& out) {
  out.fl   = (int16_t)read_mm(0);
  out.fc   = (int16_t)read_mm(1);
  out.fr   = (int16_t)read_mm(2);
  out.rear = (int16_t)read_mm(3);
  out.down = (int16_t)read_mm(4);
}

#else
// ===========================================================================
// STUB — no ToF sensors wired. Report a clear room so the reflex/zone logic stays
// in CLEAR; down=60 mm (floor present, under the cliff threshold). OBSTACLE
// AVOIDANCE IS INACTIVE in this build.
// ===========================================================================
void hal_tof_init() {}

void hal_read_tof(TofMm& out) {
  out.fl = out.fc = out.fr = out.rear = 1500;
  out.down = 60;
}
#endif  // MOTION_TOF_PRESENT
