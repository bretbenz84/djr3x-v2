// i2c_trunk.cpp — lifecycle and bounded recovery for the shared sensor bus.
#include <Arduino.h>
#include <Wire.h>

#include "i2c_trunk.h"
#include "pins.h"
#include "hal.h"
#include "calib.h"
#include "proto_io.h"

static bool s_started = false;

static void apply_settings() {
  Wire.setClock(I2C_TRUNK_CLOCK_HZ);
  Wire.setTimeOut(I2C_TRUNK_TIMEOUT_MS);
}

void i2c_trunk_init() {
  if (s_started) return;
  Wire.begin(PIN_I2C_SDA, PIN_I2C_SCL);
  apply_settings();
  s_started = true;
}

// Nine SCL pulses release a slave that lost byte alignment while holding SDA.
// Open-drain outputs preserve normal I2C electrical behavior.
static bool clear_stuck_lines() {
  pinMode(PIN_I2C_SDA, INPUT_PULLUP);
  pinMode(PIN_I2C_SCL, INPUT_PULLUP);
  delayMicroseconds(10);

  if (digitalRead(PIN_I2C_SDA) == LOW) {
    pinMode(PIN_I2C_SCL, OUTPUT_OPEN_DRAIN);
    digitalWrite(PIN_I2C_SCL, HIGH);
    for (int i = 0; i < 9 && digitalRead(PIN_I2C_SDA) == LOW; ++i) {
      digitalWrite(PIN_I2C_SCL, LOW);
      delayMicroseconds(8);
      digitalWrite(PIN_I2C_SCL, HIGH);
      delayMicroseconds(8);
    }
  }

  // Emit a STOP while both pins are still under our control.
  pinMode(PIN_I2C_SDA, OUTPUT_OPEN_DRAIN);
  digitalWrite(PIN_I2C_SDA, LOW);
  delayMicroseconds(8);
  pinMode(PIN_I2C_SCL, OUTPUT_OPEN_DRAIN);
  digitalWrite(PIN_I2C_SCL, HIGH);
  delayMicroseconds(8);
  digitalWrite(PIN_I2C_SDA, HIGH);
  delayMicroseconds(8);

  pinMode(PIN_I2C_SDA, INPUT_PULLUP);
  pinMode(PIN_I2C_SCL, INPUT_PULLUP);
  return digitalRead(PIN_I2C_SDA) == HIGH && digitalRead(PIN_I2C_SCL) == HIGH;
}

bool i2c_trunk_recover(const char* reason) {
  Wire.end();
  delay(2);
  const bool lines_released = clear_stuck_lines();
  Wire.begin(PIN_I2C_SDA, PIN_I2C_SCL);
  apply_settings();
  s_started = true;

#if MOTION_TOF_PRESENT && MOTION_TOF_USE_MUX
  // Recovery must not leave an unknown downstream branch connected. Each ToF
  // read explicitly selects its own channel again.
  Wire.beginTransmission(TOF_MUX_ADDR);
  Wire.write((uint8_t)0x00);
  const uint8_t mux_rc = Wire.endTransmission();
#else
  const uint8_t mux_rc = 0;
#endif

  char msg[144];
  snprintf(msg, sizeof(msg),
           "i2c: trunk recovered (%s; lines=%s, mux_deselect_rc=%u)",
           reason ? reason : "unspecified", lines_released ? "high" : "stuck",
           (unsigned)mux_rc);
  emit_log(lines_released && mux_rc == 0 ? "warn" : "error", msg);
  return lines_released && mux_rc == 0;
}
