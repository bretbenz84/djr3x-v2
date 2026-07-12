// battery.cpp — INA226 pack monitor (see battery.h for wiring).
//
// Minimal register-level driver (no library dependency):
//   0x00 config      — set averaging + conversion times, continuous mode
//   0x02 bus voltage — 1.25 mV/LSB, measured at the VBUS pin (up to 36V: covers
//                      the LiFePO4 charger's 14.6V peak with no divider)
//   0x01 shunt volt  — 2.5 uV/LSB (only used when BATT_SHUNT_MICROOHM > 0)
//   0xFE mfr id      — 0x5449 ("TI"), used as the presence probe
//
// Reads are smoothed with an EMA so telemetry doesn't jitter; the LiFePO4 SOC
// interpretation happens on the HOST (intelligence side), not here.

#include <Arduino.h>
#include <Wire.h>
#include "battery.h"
#include "context.h"
#include "pins.h"

#ifndef BATT_INA226_ADDR
#define BATT_INA226_ADDR 0x40      // A0/A1 to GND (module default)
#endif
#ifndef BATT_SHUNT_MICROOHM
#define BATT_SHUNT_MICROOHM 0      // 0 = voltage-only (stock 100 mOhm module
                                   // shunt maxes at +/-0.8A — not motor-ranged)
#endif

static bool  s_present = false;
static float s_mv_ema  = -1.0f;
static float s_ma_ema  = 0.0f;

static const uint16_t REG_CONFIG = 0x00;
static const uint16_t REG_SHUNT  = 0x01;
static const uint16_t REG_BUS    = 0x02;
static const uint16_t REG_MFR_ID = 0xFE;

static bool ina_write16(uint8_t reg, uint16_t val) {
  Wire.beginTransmission(BATT_INA226_ADDR);
  Wire.write(reg);
  Wire.write((uint8_t)(val >> 8));
  Wire.write((uint8_t)(val & 0xFF));
  return Wire.endTransmission() == 0;
}

static bool ina_read16(uint8_t reg, uint16_t &out) {
  Wire.beginTransmission(BATT_INA226_ADDR);
  Wire.write(reg);
  if (Wire.endTransmission(false) != 0) return false;
  if (Wire.requestFrom((int)BATT_INA226_ADDR, 2) != 2) return false;
  out = ((uint16_t)Wire.read() << 8) | Wire.read();
  return true;
}

void battery_init() {
  // Wire is normally begun by tof_init(); begin() again is harmless and covers
  // ToF-less bench boards.
  Wire.begin(PIN_I2C_SDA, PIN_I2C_SCL);

  uint16_t mfr = 0;
  s_present = ina_read16(REG_MFR_ID, mfr) && mfr == 0x5449;
  if (!s_present) {
    Serial.println("{\"v\":1,\"type\":\"log\",\"msg\":\"battery: no INA226 at 0x40 — batt_mv=-1 (unknown)\"}");
    return;
  }
  // AVG=16 samples, 1.1ms conversions, shunt+bus continuous: ~35ms per averaged
  // result — far faster than the 1 Hz tick, so every read is fresh.
  ina_write16(REG_CONFIG, 0x4527);
  Serial.println("{\"v\":1,\"type\":\"log\",\"msg\":\"battery: INA226 online (pack voltage sense)\"}");
}

bool battery_present() { return s_present; }

void battery_tick() {
  if (!s_present) return;

  // Voltage and current are INDEPENDENT measurements (VBUS pin vs the IN+/IN-
  // shunt path) — a build can have either wired without the other, so an
  // unwired VBUS must not block the current read (it used to early-return here).
  bool have_mv = false;
  uint16_t raw = 0;
  if (ina_read16(REG_BUS, raw)) {          // transient bus error: keep last EMA
    float mv = raw * 1.25f;
    if (mv >= 1000.0f) {                   // VBUS unwired/floating — don't report garbage
      s_mv_ema = (s_mv_ema < 0.0f) ? mv : (0.8f * s_mv_ema + 0.2f * mv);
      have_mv = true;
    }
  }

#if BATT_SHUNT_MICROOHM > 0
  uint16_t sraw = 0;
  if (ina_read16(REG_SHUNT, sraw)) {
    // 2.5 uV/LSB across the shunt; I = V/R. Signed register.
    float uv = (int16_t)sraw * 2.5f;
    float ma = uv * 1000.0f / (float)BATT_SHUNT_MICROOHM;
    s_ma_ema = 0.8f * s_ma_ema + 0.2f * ma;
  }
#endif

  LOCK_STATE();
  if (have_mv) g_ctx.batt_mv = (int16_t)(s_mv_ema + 0.5f);
#if BATT_SHUNT_MICROOHM > 0
  g_ctx.batt_ma = (int16_t)s_ma_ema;
#endif
  UNLOCK_STATE();
}
