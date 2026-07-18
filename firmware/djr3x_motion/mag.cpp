// mag.cpp — QMC5883L driver (see mag.h for the role and the QMC-vs-HMC warning).
//
// Register-level, no library dependency (mirrors imu.cpp/env.cpp):
//   0x00..0x05  data: X LSB,MSB / Y LSB,MSB / Z LSB,MSB (int16 little-endian)
//   0x06        status: bit0 DRDY, bit1 OVL (overflow), bit2 DOR (skipped)
//   0x09        control1: OSR[7:6] RNG[5:4] ODR[3:2] MODE[1:0]
//   0x0A        control2: 0x80 = soft reset
//   0x0B        SET/RESET period — datasheet says write 0x01
//   0x0D        chip id, always 0xFF on a genuine QMC5883L
//
// Config chosen for a magnetically hostile platform: ±8 G range (motors,
// BTS7960s, LED runs — ±2 G would clip near the drivetrain), 50 Hz ODR,
// OSR 512 (max filtering). 8 G = 3000 LSB/gauss; the Mac works in raw counts
// and calibrates the scale away, so the LSB constant is documentation only.
#include <Arduino.h>
#include <Wire.h>
#include "mag.h"
#include "context.h"
#include "pins.h"
#include "proto_io.h"   // emit_log — bring-up diagnostics

#ifndef MAG_QMC_ADDR
#define MAG_QMC_ADDR 0x0D           // fixed on the QMC5883L (no strap options)
#endif

static bool    s_present = false;
static uint8_t s_err_streak = 0;

// Same trunk-protection policy as imu.cpp/env.cpp: a sensor that degrades
// mid-session burns an I2C timeout per tick and disturbs the shared bus —
// self-disable after a solid failure streak (re-probed at next boot).
static const uint8_t MAG_ERR_STREAK_DISABLE = 30;   // ~3 s at the 10 Hz tick

static bool mag_write8(uint8_t reg, uint8_t val) {
  Wire.beginTransmission(MAG_QMC_ADDR);
  Wire.write(reg);
  Wire.write(val);
  return Wire.endTransmission() == 0;
}

static bool mag_read(uint8_t reg, uint8_t* buf, uint8_t n) {
  Wire.beginTransmission(MAG_QMC_ADDR);
  Wire.write(reg);
  if (Wire.endTransmission(false) != 0) return false;
  if (Wire.requestFrom((int)MAG_QMC_ADDR, (int)n) != n) return false;
  for (uint8_t i = 0; i < n; i++) buf[i] = Wire.read();
  return true;
}

void mag_init() {
  Wire.begin(PIN_I2C_SDA, PIN_I2C_SCL);   // harmless if already begun (trunk peers)

  uint8_t id = 0;
  if (!mag_read(0x0D, &id, 1) || id != 0xFF) {
    emit_log("info", "mag: no QMC5883L at 0x0D — compass unavailable");
    return;
  }

  mag_write8(0x0A, 0x80);                 // soft reset
  delay(10);
  mag_write8(0x0B, 0x01);                 // SET/RESET period (datasheet-required)
  // OSR=512 (00), RNG=8G (01), ODR=50Hz (10), MODE=continuous (01) -> 0x19
  if (!mag_write8(0x09, 0x19)) {
    emit_log("warn", "mag: QMC5883L configure failed — compass unavailable");
    return;
  }

  s_present = true;
  emit_log("info", "mag: QMC5883L online at 0x0D (raw axes -> telemetry; host fuses)");
}

bool mag_present() { return s_present; }

void mag_tick() {
  if (!s_present) return;

  uint8_t st = 0;
  if (!mag_read(0x06, &st, 1)) {
    if (s_err_streak < 255) s_err_streak++;
    if (s_err_streak == MAG_ERR_STREAK_DISABLE) {
      s_present = false;
      LOCK_STATE();
      g_ctx.mag.ok = false;
      UNLOCK_STATE();
      emit_log("warn", "mag: QMC5883L disabled after repeated bus errors — check its wiring");
    }
    return;
  }
  s_err_streak = 0;
  if (!(st & 0x01)) return;               // no fresh sample yet (DRDY clear)

  uint8_t b[6];
  if (!mag_read(0x00, b, 6)) return;      // transient: keep last published sample
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
