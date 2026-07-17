// env.cpp — Bosch BMP280/BME280 driver (see env.h for the role).
//
// Register-level, no library dependency (mirrors imu.cpp/battery.cpp):
//   0xD0 CHIP_ID     — 0x58 = BMP280 (temp+pressure), 0x60 = BME280 (+humidity).
//                      The GY-BME280 breakouts ship with EITHER chip under a
//                      shared "BME/BMP280" silkscreen — only this register knows.
//   0x88..0xA1       — temp/pressure calibration (dig_T*, dig_P*), little-endian
//   0xA1, 0xE1..0xE7 — humidity calibration (dig_H1..H6, BME280 only)
//   0xF2 CTRL_HUM    — humidity oversampling x1 (BME280 only; write BEFORE CTRL_MEAS)
//   0xF4 CTRL_MEAS   — temp x1, pressure x1, normal mode
//   0xF5 CONFIG      — standby 1000 ms, IIR filter off (slow, low-power sampling)
//   0xF7..0xFE       — burst read: pressure(3) temp(3) humidity(2)
// Compensation formulas are Bosch's reference integer implementations
// (datasheet §4.2.3 / §8.1), verbatim including the 32/64-bit fixed-point types.
#include <Arduino.h>
#include <Wire.h>
#include "env.h"
#include "context.h"
#include "pins.h"
#include "proto_io.h"   // emit_log — bring-up diagnostics

static uint8_t s_addr = 0;          // 0x76/0x77 once probed, 0 = not present
static bool    s_has_humidity = false;   // chip ID 0x60 (BME280)
static uint8_t s_err_streak = 0;

// Self-disable after solid failures, same trunk-protection policy as imu.cpp:
// a degraded sensor burning an I2C timeout on every tick disturbs the shared bus.
static const uint8_t ENV_ERR_STREAK_DISABLE = 10;   // ~20 s at the 2 s tick

// Factory calibration (names per datasheet).
static uint16_t dig_T1;
static int16_t  dig_T2, dig_T3;
static uint16_t dig_P1;
static int16_t  dig_P2, dig_P3, dig_P4, dig_P5, dig_P6, dig_P7, dig_P8, dig_P9;
static uint8_t  dig_H1, dig_H3;
static int16_t  dig_H2, dig_H4, dig_H5;
static int8_t   dig_H6;
static int32_t  t_fine;             // shared between temp and pressure/humidity comp

static bool env_write8(uint8_t reg, uint8_t val) {
  Wire.beginTransmission(s_addr);
  Wire.write(reg);
  Wire.write(val);
  return Wire.endTransmission() == 0;
}

static bool env_read(uint8_t reg, uint8_t* buf, uint8_t n) {
  Wire.beginTransmission(s_addr);
  Wire.write(reg);
  if (Wire.endTransmission(false) != 0) return false;
  if (Wire.requestFrom((int)s_addr, (int)n) != n) return false;
  for (uint8_t i = 0; i < n; i++) buf[i] = Wire.read();
  return true;
}

// Bosch reference compensation (datasheet). Raw ADC in, physical units out.
static float comp_temp_c(int32_t adc_T) {
  int32_t var1 = ((((adc_T >> 3) - ((int32_t)dig_T1 << 1))) * ((int32_t)dig_T2)) >> 11;
  int32_t var2 = (((((adc_T >> 4) - ((int32_t)dig_T1)) *
                    ((adc_T >> 4) - ((int32_t)dig_T1))) >> 12) * ((int32_t)dig_T3)) >> 14;
  t_fine = var1 + var2;
  return ((t_fine * 5 + 128) >> 8) / 100.0f;
}

static float comp_pressure_hpa(int32_t adc_P) {
  int64_t var1 = ((int64_t)t_fine) - 128000;
  int64_t var2 = var1 * var1 * (int64_t)dig_P6;
  var2 += ((var1 * (int64_t)dig_P5) << 17);
  var2 += (((int64_t)dig_P4) << 35);
  var1 = ((var1 * var1 * (int64_t)dig_P3) >> 8) + ((var1 * (int64_t)dig_P2) << 12);
  var1 = (((((int64_t)1) << 47) + var1)) * ((int64_t)dig_P1) >> 33;
  if (var1 == 0) return 0.0f;                       // avoid divide-by-zero
  int64_t p = 1048576 - adc_P;
  p = (((p << 31) - var2) * 3125) / var1;
  var1 = (((int64_t)dig_P9) * (p >> 13) * (p >> 13)) >> 25;
  var2 = (((int64_t)dig_P8) * p) >> 19;
  p = ((p + var1 + var2) >> 8) + (((int64_t)dig_P7) << 4);
  return (float)p / 256.0f / 100.0f;                // Pa(Q24.8) -> hPa
}

static float comp_humidity_pct(int32_t adc_H) {
  int32_t v = t_fine - ((int32_t)76800);
  v = (((((adc_H << 14) - (((int32_t)dig_H4) << 20) - (((int32_t)dig_H5) * v)) +
         ((int32_t)16384)) >> 15) *
       (((((((v * ((int32_t)dig_H6)) >> 10) *
            (((v * ((int32_t)dig_H3)) >> 11) + ((int32_t)32768))) >> 10) +
          ((int32_t)2097152)) * ((int32_t)dig_H2) + 8192) >> 14));
  v = v - (((((v >> 15) * (v >> 15)) >> 7) * ((int32_t)dig_H1)) >> 4);
  if (v < 0) v = 0;
  if (v > 419430400) v = 419430400;
  return (float)(v >> 12) / 1024.0f;
}

static bool load_calibration() {
  uint8_t b[26];
  if (!env_read(0x88, b, 26)) return false;         // 0x88..0xA1 (temp+press+H1)
  dig_T1 = (uint16_t)(b[0] | (b[1] << 8));
  dig_T2 = (int16_t)(b[2] | (b[3] << 8));
  dig_T3 = (int16_t)(b[4] | (b[5] << 8));
  dig_P1 = (uint16_t)(b[6] | (b[7] << 8));
  dig_P2 = (int16_t)(b[8]  | (b[9]  << 8));
  dig_P3 = (int16_t)(b[10] | (b[11] << 8));
  dig_P4 = (int16_t)(b[12] | (b[13] << 8));
  dig_P5 = (int16_t)(b[14] | (b[15] << 8));
  dig_P6 = (int16_t)(b[16] | (b[17] << 8));
  dig_P7 = (int16_t)(b[18] | (b[19] << 8));
  dig_P8 = (int16_t)(b[20] | (b[21] << 8));
  dig_P9 = (int16_t)(b[22] | (b[23] << 8));
  if (s_has_humidity) {
    dig_H1 = b[25];                                 // 0xA1
    uint8_t h[7];
    if (!env_read(0xE1, h, 7)) return false;        // 0xE1..0xE7
    dig_H2 = (int16_t)(h[0] | (h[1] << 8));
    dig_H3 = h[2];
    dig_H4 = (int16_t)((h[3] << 4) | (h[4] & 0x0F));
    dig_H5 = (int16_t)((h[5] << 4) | (h[4] >> 4));
    dig_H6 = (int8_t)h[6];
  }
  return true;
}

void env_init() {
  Wire.begin(PIN_I2C_SDA, PIN_I2C_SCL);   // harmless if already begun (tof/battery/imu)

  // Probe 0x76 (SDO low, breakout default) then 0x77 (SDO high).
  uint8_t id = 0;
  for (uint8_t addr : {(uint8_t)0x76, (uint8_t)0x77}) {
    s_addr = addr;
    if (env_read(0xD0, &id, 1) && (id == 0x58 || id == 0x60)) break;
    s_addr = 0;
  }
  if (s_addr == 0) {
    emit_log("info", "env: no BMP280/BME280 at 0x76/0x77 — climate unavailable");
    return;
  }
  s_has_humidity = (id == 0x60);

  if (!load_calibration()) {
    emit_log("warn", "env: calibration read failed — climate unavailable");
    s_addr = 0;
    return;
  }

  // Config BEFORE ctrl_meas per datasheet ordering; ctrl_hum only latches on a
  // subsequent ctrl_meas write, so keep this exact sequence.
  env_write8(0xF5, 0xA0);                  // standby 1000 ms, IIR off, SPI off
  if (s_has_humidity) env_write8(0xF2, 0x01);  // humidity oversampling x1
  if (!env_write8(0xF4, 0x27)) {           // temp x1, press x1, NORMAL mode
    emit_log("warn", "env: sensor configure failed — climate unavailable");
    s_addr = 0;
    return;
  }

  char buf[80];
  snprintf(buf, sizeof(buf), "env: %s online at 0x%02X (%s)",
           s_has_humidity ? "BME280" : "BMP280", s_addr,
           s_has_humidity ? "temp/pressure/humidity" : "temp/pressure — no humidity on this chip");
  emit_log("info", buf);
}

bool env_present() { return s_addr != 0; }

void env_tick() {
  if (s_addr == 0) return;

  uint8_t b[8];
  if (!env_read(0xF7, b, 8)) {
    if (s_err_streak < 255) s_err_streak++;
    if (s_err_streak == ENV_ERR_STREAK_DISABLE) {
      s_addr = 0;
      LOCK_STATE();
      g_ctx.env.ok = false;
      UNLOCK_STATE();
      emit_log("warn", "env: sensor disabled after repeated bus errors — check its wiring");
    }
    return;
  }
  s_err_streak = 0;

  const int32_t adc_P = ((int32_t)b[0] << 12) | ((int32_t)b[1] << 4) | (b[2] >> 4);
  const int32_t adc_T = ((int32_t)b[3] << 12) | ((int32_t)b[4] << 4) | (b[5] >> 4);
  const int32_t adc_H = ((int32_t)b[6] << 8) | b[7];

  // comp_temp_c MUST run first: it sets t_fine for the other two.
  const float temp = comp_temp_c(adc_T);
  const float hpa  = comp_pressure_hpa(adc_P);
  const float rh   = s_has_humidity ? comp_humidity_pct(adc_H) : -1.0f;

  LOCK_STATE();
  g_ctx.env.ok     = true;
  g_ctx.env.temp_c = temp;
  g_ctx.env.hpa    = hpa;
  g_ctx.env.rh     = rh;          // -1 = chip has no humidity sensor
  UNLOCK_STATE();
}
