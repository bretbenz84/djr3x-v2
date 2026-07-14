// imu.cpp — MPU-6050 driver + complementary attitude filter (see imu.h).
//
// Minimal register-level driver (no library dependency, mirrors battery.cpp):
//   0x6B PWR_MGMT_1  — 0x80 reset, then 0x01 wake with PLL-X clock (datasheet-
//                      recommended over the default 8 MHz RC for gyro stability)
//   0x1A CONFIG      — DLPF 3: ~44 Hz accel / 42 Hz gyro low-pass (kills motor vibe)
//   0x1B GYRO_CONFIG — FS 0: ±250 °/s, 131 LSB/(°/s)
//   0x1C ACCEL_CONFIG— FS 0: ±2 g, 16384 LSB/g
//   0x3B..0x48       — accel xyz, temp, gyro xyz (14-byte burst read)
//   0x75 WHO_AM_I    — 0x68 (clones report 0x72/0x98; accepted with a log note)
//
// Attitude: complementary filter — gyro integration for short-term truth, the
// accel gravity vector pulling pitch/roll back long-term. Yaw has no gravity
// reference (no magnetometer indoors BY DESIGN, roadmap §5), so it's pure
// bias-corrected gyro integration relative to boot heading: expect slow drift.
// Gyro bias is measured at boot over ~1 s of stillness (the base boots idle).
#include <Arduino.h>
#include <Wire.h>
#include <math.h>
#include "imu.h"
#include "context.h"
#include "pins.h"
#include "proto_io.h"   // emit_log — bring-up diagnostics

#ifndef IMU_MPU_ADDR
#define IMU_MPU_ADDR 0x68           // AD0 low (GY-521 default)
#endif

static bool  s_present = false;
static float s_bias_gx = 0, s_bias_gy = 0, s_bias_gz = 0;  // deg/s at rest
static float s_pitch = 0, s_roll = 0, s_yaw = 0;           // deg, filtered
static uint8_t s_err_streak = 0;   // consecutive failed samples (self-disable below)

// A sensor that passes the boot probe but then degrades mid-session (field case:
// loosening jumpers) fails EVERY 20 ms sample, and each failure burns an I2C
// timeout on the shared trunk — disturbing the ToF/INA reads and starving lower-
// priority tasks. After this many consecutive failures the IMU self-disables for
// the session (re-probed at next boot); attitude reads ok:false, honestly.
static const uint8_t IMU_ERR_STREAK_DISABLE = 25;   // ~0.5 s of solid failures

static const float ACCEL_LSB_PER_G   = 16384.0f;  // FS ±2 g
static const float GYRO_LSB_PER_DPS  = 131.0f;    // FS ±250 °/s

static bool mpu_write8(uint8_t reg, uint8_t val) {
  Wire.beginTransmission(IMU_MPU_ADDR);
  Wire.write(reg);
  Wire.write(val);
  return Wire.endTransmission() == 0;
}

static bool mpu_read(uint8_t reg, uint8_t* buf, uint8_t n) {
  Wire.beginTransmission(IMU_MPU_ADDR);
  Wire.write(reg);
  if (Wire.endTransmission(false) != 0) return false;
  if (Wire.requestFrom((int)IMU_MPU_ADDR, (int)n) != n) return false;
  for (uint8_t i = 0; i < n; i++) buf[i] = Wire.read();
  return true;
}

// One burst sample in physical units (chip frame). Returns false on bus error.
static bool mpu_sample(float& ax, float& ay, float& az,
                       float& gx, float& gy, float& gz) {
  uint8_t b[14];
  if (!mpu_read(0x3B, b, 14)) return false;
  ax = (int16_t)((b[0]  << 8) | b[1])  / ACCEL_LSB_PER_G;
  ay = (int16_t)((b[2]  << 8) | b[3])  / ACCEL_LSB_PER_G;
  az = (int16_t)((b[4]  << 8) | b[5])  / ACCEL_LSB_PER_G;
  // b[6..7] = temp, unused
  gx = (int16_t)((b[8]  << 8) | b[9])  / GYRO_LSB_PER_DPS;
  gy = (int16_t)((b[10] << 8) | b[11]) / GYRO_LSB_PER_DPS;
  gz = (int16_t)((b[12] << 8) | b[13]) / GYRO_LSB_PER_DPS;
  return true;
}

// Pitch/roll of the gravity vector (deg), chip frame.
static inline float accel_pitch(float ax, float ay, float az) {
  return atan2f(-ax, sqrtf(ay * ay + az * az)) * 180.0f / (float)M_PI;
}
static inline float accel_roll(float ax, float ay, float az) {
  (void)ax;
  return atan2f(ay, az) * 180.0f / (float)M_PI;
}

void imu_init() {
  Wire.begin(PIN_I2C_SDA, PIN_I2C_SCL);   // harmless if already begun (tof/battery)

  uint8_t who = 0;
  if (!mpu_read(0x75, &who, 1)) {
    emit_log("info", "imu: no MPU-6050 at 0x68 — attitude unavailable");
    return;
  }
  // Genuine parts report 0x68; common clones report 0x72/0x98 and work fine.
  char buf[72];
  if (who != 0x68) {
    snprintf(buf, sizeof(buf), "imu: WHO_AM_I=0x%02X (clone?) — continuing", who);
    emit_log("warn", buf);
  }

  // Reset, wake with PLL-X clock, configure filters + full-scale ranges.
  mpu_write8(0x6B, 0x80); delay(100);     // DEVICE_RESET
  if (!mpu_write8(0x6B, 0x01)) {          // wake, CLKSEL=PLL X gyro
    emit_log("warn", "imu: MPU-6050 wake failed — attitude unavailable");
    return;
  }
  mpu_write8(0x1A, 0x03);                 // DLPF 44/42 Hz
  mpu_write8(0x1B, 0x00);                 // gyro ±250 °/s
  mpu_write8(0x1C, 0x00);                 // accel ±2 g
  delay(50);                              // let the DLPF settle

  // Gyro bias calibration: average ~1 s of samples at rest (boot is idle by
  // design — motors start disabled). A moving boot just yields a worse bias;
  // it self-reports via the spread check below.
  float sgx = 0, sgy = 0, sgz = 0, ax, ay, az, gx, gy, gz;
  int   n = 0;
  float gmin = 1e9f, gmax = -1e9f;
  for (int i = 0; i < 100; i++) {
    if (mpu_sample(ax, ay, az, gx, gy, gz)) {
      sgx += gx; sgy += gy; sgz += gz; n++;
      if (gz < gmin) gmin = gz;
      if (gz > gmax) gmax = gz;
    }
    delay(10);
  }
  if (n < 50) {
    emit_log("warn", "imu: MPU-6050 bias calibration failed (bus errors) — attitude unavailable");
    return;
  }
  s_bias_gx = sgx / n; s_bias_gy = sgy / n; s_bias_gz = sgz / n;

  // Seed attitude from the gravity vector so the filter starts converged.
  if (mpu_sample(ax, ay, az, gx, gy, gz)) {
    s_pitch = accel_pitch(ax, ay, az);
    s_roll  = accel_roll(ax, ay, az);
  }
  s_yaw = 0.0f;                            // yaw is relative to boot heading
  s_present = true;

  snprintf(buf, sizeof(buf), "imu: MPU-6050 online (gyro bias z %+.2f dps, spread %.2f)",
           s_bias_gz, gmax - gmin);
  emit_log("info", buf);
}

bool imu_present() { return s_present; }

void imu_tick(float dt) {
  if (!s_present || dt <= 0.0f) return;

  float ax, ay, az, gx, gy, gz;
  if (!mpu_sample(ax, ay, az, gx, gy, gz)) {
    // Transient bus error: keep the last attitude. A SOLID failure streak means
    // the sensor is gone/degraded mid-session — self-disable so its timeouts stop
    // disturbing the shared I2C trunk (see IMU_ERR_STREAK_DISABLE above).
    if (s_err_streak < 255) s_err_streak++;
    if (s_err_streak == IMU_ERR_STREAK_DISABLE) {
      s_present = false;
      LOCK_STATE();
      g_ctx.imu.ok = false;
      UNLOCK_STATE();
      emit_log("warn", "imu: MPU-6050 disabled after repeated bus errors — check its wiring");
    }
    return;
  }
  s_err_streak = 0;

  gx -= s_bias_gx; gy -= s_bias_gy; gz -= s_bias_gz;

  // Complementary filter: gyro short-term, accel gravity long-term. 0.98 at
  // 50 Hz ≈ 1 s time constant — slow enough that drive accelerations don't
  // tilt the estimate, fast enough to null gyro drift in pitch/roll.
  const float ap = accel_pitch(ax, ay, az);
  const float ar = accel_roll(ax, ay, az);
  s_pitch = 0.98f * (s_pitch + gy * dt) + 0.02f * ap;
  s_roll  = 0.98f * (s_roll  + gx * dt) + 0.02f * ar;

  // Yaw: bias-corrected integration, wrapped to (-180, 180]. Drifts slowly —
  // display/diagnostic grade until fused with odometry (roadmap Phase A step 2).
  s_yaw += gz * dt;
  while (s_yaw >  180.0f) s_yaw -= 360.0f;
  while (s_yaw <= -180.0f) s_yaw += 360.0f;

  LOCK_STATE();
  g_ctx.imu.ok    = true;
  g_ctx.imu.pitch = s_pitch;
  g_ctx.imu.roll  = s_roll;
  g_ctx.imu.yaw   = s_yaw;
  UNLOCK_STATE();
}
