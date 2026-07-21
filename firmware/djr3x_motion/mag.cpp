// mag.cpp — QMC5883 magnetometer driver (raw axes -> telemetry; host fuses).
//
// TWO different chips, auto-detected (their register maps are NOT compatible):
//   • QMC5883L @ 0x0D — chip-id 0xFF @reg 0x0D; data @0x00; status @0x06.
//   • QMC5883P @ 0x2C — chip-id 0x80 @reg 0x00; data @0x01; status @0x09.
// GY-271 boards silkscreened "QMC5883L" very often actually carry the P-variant
// at 0x2C (field lesson: this build's board did). Both publish RAW int16 counts
// at 10 Hz in the `mag` telemetry block; hard/soft-iron calibration, tilt
// compensation, and current-gated fusion all live on the Mac (hardware/compass.py).
//
// Register-level, no library dependency (mirrors imu.cpp/env.cpp).
// Config chosen for a magnetically hostile platform: ±8 G range (motors,
// BTS7960s, LED runs — a narrow range clips near the drivetrain). The Mac works
// in raw counts and calibrates the scale away, so the range is documentation.
#include <Arduino.h>
#include <Wire.h>
#include "mag.h"
#include "context.h"
#include "pins.h"
#include "proto_io.h"   // emit_log — bring-up diagnostics

#define MAG_L_ADDR 0x0D            // QMC5883L (fixed, no strap options)
#define MAG_P_ADDR 0x2C            // QMC5883P (fixed)

enum MagVariant { MAG_NONE = 0, MAG_L, MAG_P };
static MagVariant s_variant    = MAG_NONE;
static uint8_t    s_addr       = 0;
static uint8_t    s_data_reg   = 0;   // burst base for X/Y/Z (6 bytes, int16 LE)
static uint8_t    s_status_reg = 0;   // DRDY = bit0, OVL = bit1 on BOTH variants
static uint8_t    s_err_streak = 0;

// Same trunk-protection policy as imu.cpp/env.cpp: a sensor that degrades
// mid-session burns an I2C timeout per tick and disturbs the shared bus —
// self-disable after a solid failure streak (re-probed at next boot).
static const uint8_t MAG_ERR_STREAK_DISABLE = 30;   // ~3 s at the 10 Hz tick

static bool mag_write8(uint8_t addr, uint8_t reg, uint8_t val) {
  Wire.beginTransmission(addr);
  Wire.write(reg);
  Wire.write(val);
  return Wire.endTransmission() == 0;
}

static bool mag_read(uint8_t addr, uint8_t reg, uint8_t* buf, uint8_t n) {
  Wire.beginTransmission(addr);
  Wire.write(reg);
  if (Wire.endTransmission(false) != 0) return false;
  if (Wire.requestFrom((int)addr, (int)n) != n) return false;
  for (uint8_t i = 0; i < n; i++) buf[i] = Wire.read();
  return true;
}

// QMC5883L: soft reset, SET/RESET period, then OSR512 / ±8G / 50Hz / continuous.
static bool init_qmc_l() {
  uint8_t id = 0;
  if (!mag_read(MAG_L_ADDR, 0x0D, &id, 1) || id != 0xFF) return false;
  mag_write8(MAG_L_ADDR, 0x0A, 0x80);           // soft reset
  delay(10);
  mag_write8(MAG_L_ADDR, 0x0B, 0x01);           // SET/RESET period (datasheet-required)
  // OSR=512 (00), RNG=8G (01), ODR=50Hz (10), MODE=continuous (01) -> 0x19
  if (!mag_write8(MAG_L_ADDR, 0x09, 0x19)) return false;
  s_variant = MAG_L; s_addr = MAG_L_ADDR; s_data_reg = 0x00; s_status_reg = 0x06;
  return true;
}

// QMC5883P: CR2 soft-reset, axis-sign config (datasheet-required), ±8G range,
// then CR1 = Normal(continuous)/100Hz/8x oversample/8x downsample.
static bool init_qmc_p() {
  uint8_t id = 0;
  if (!mag_read(MAG_P_ADDR, 0x00, &id, 1) || id != 0x80) return false;
  mag_write8(MAG_P_ADDR, 0x0B, 0x80);           // CR2: soft reset
  delay(10);
  mag_write8(MAG_P_ADDR, 0x29, 0x06);           // axis sign define (datasheet-required)
  mag_write8(MAG_P_ADDR, 0x0B, 0x08);           // CR2: range = ±8 G (field 0x2 << 2)
  // CR1 (0x0A): DOWN_SMPL8 (0xC0) | OVR_SMPL8 (0x00) | ODR_100HZ (0x08) | MODE_NORMAL (0x01)
  if (!mag_write8(MAG_P_ADDR, 0x0A, 0xC9)) return false;
  s_variant = MAG_P; s_addr = MAG_P_ADDR; s_data_reg = 0x01; s_status_reg = 0x09;
  return true;
}

void mag_init() {
  if (init_qmc_l()) {
    emit_log("info", "mag: QMC5883L online at 0x0D (raw axes -> telemetry; host fuses)");
    return;
  }
  if (init_qmc_p()) {
    emit_log("info", "mag: QMC5883P online at 0x2C (raw axes -> telemetry; host fuses)");
    return;
  }
  emit_log("info", "mag: no QMC5883L/P at 0x0D/0x2C — compass unavailable");
}

bool mag_present() { return s_variant != MAG_NONE; }

void mag_tick() {
  if (s_variant == MAG_NONE) return;

  uint8_t st = 0;
  if (!mag_read(s_addr, s_status_reg, &st, 1)) {
    if (s_err_streak < 255) s_err_streak++;
    if (s_err_streak == MAG_ERR_STREAK_DISABLE) {
      s_variant = MAG_NONE;
      LOCK_STATE();
      g_ctx.mag.ok = false;
      UNLOCK_STATE();
      emit_log("warn", "mag: QMC5883 disabled after repeated bus errors — check its wiring");
    }
    return;
  }
  s_err_streak = 0;
  if (!(st & 0x01)) return;               // no fresh sample yet (DRDY clear)

  uint8_t b[6];
  if (!mag_read(s_addr, s_data_reg, b, 6)) return;   // transient: keep last sample
  const int16_t x = (int16_t)(b[0] | (b[1] << 8));
  const int16_t y = (int16_t)(b[2] | (b[3] << 8));
  const int16_t z = (int16_t)(b[4] | (b[5] << 8));

  LOCK_STATE();
  g_ctx.mag.ok = true;
  g_ctx.mag.x = x;                        // RAW counts — calibration is host-side
  g_ctx.mag.y = y;
  g_ctx.mag.z = z;
  g_ctx.mag.ovl = (st & 0x02) != 0;       // field overflow — host should reject sample
  UNLOCK_STATE();
}
