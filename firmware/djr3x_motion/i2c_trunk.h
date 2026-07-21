// i2c_trunk.h — single owner for the shared GPIO21/22 I2C controller.
#pragma once

#include <Arduino.h>

// Call once during setup, before any trunk client probes its device.
void i2c_trunk_init();

// Recover the ESP32 controller after a sustained transaction failure. This
// clocks a potentially-stuck slave free, restarts Wire with the canonical
// settings, and deselects every TCA9548A channel. Runtime trunk access is
// serialized on sensorTask, so callers must stay on that task.
bool i2c_trunk_recover(const char* reason);

static const uint32_t I2C_TRUNK_CLOCK_HZ = 100000;
static const uint16_t I2C_TRUNK_TIMEOUT_MS = 20;
