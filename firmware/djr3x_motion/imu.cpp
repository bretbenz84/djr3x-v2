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
#include "i2c_trunk.h"

#ifndef IMU_MPU_ADDR
#define IMU_MPU_ADDR 0x68           // AD0 low (GY-521 default)
#endif

static bool  s_present = false;
static float s_bias_gx = 0, s_bias_gy = 0, s_bias_gz = 0;  // deg/s at rest
static float s_pitch = 0, s_roll = 0, s_yaw = 0;           // deg, filtered
static uint8_t s_err_streak = 0;   // consecutive failed samples (recovery threshold below)
static uint32_t s_next_reprobe_ms = 0;
static uint8_t s_reprobe_failures = 0;
static bool s_ever_online = false;

// A sensor that passes the boot probe but then degrades mid-session (field case:
// loosening jumpers) fails EVERY 20 ms sample, and each failure burns an I2C
// timeout on the shared trunk — disturbing the ToF/INA reads and starving lower-
// priority tasks. After this many consecutive failures, recover the controller;
// if that fails, publish ok:false and enter a bounded periodic reprobe cadence.
static const uint8_t IMU_ERR_STREAK_RECOVER = 25;   // ~0.5 s of solid failures
static const uint32_t IMU_REPROBE_INTERVAL_MS = 5000;
static const uint8_t IMU_REPROBE_RECOVER_EVERY = 6; // bus reset at most every 30 s

static const float ACCEL_LSB_PER_G   = 16384.0f;  // FS ±2 g
static const float GYRO_LSB_PER_DPS  = 131.0f;    // FS ±250 °/s

enum MpuIoStage : uint8_t { MPU_IO_NONE, MPU_IO_ADDR_WRITE, MPU_IO_DATA_READ };
struct MpuIoError {
  MpuIoStage stage = MPU_IO_NONE;
  uint8_t wire_rc = 0;
  uint8_t expected = 0;
  uint8_t received = 0;
  bool used_stop = false;
};
static MpuIoError s_last_io;
static bool s_stop_fallback_logged = false;

static void set_io_error(MpuIoStage stage = MPU_IO_NONE, uint8_t wire_rc = 0,
                         uint8_t expected = 0, uint8_t received = 0,
                         bool used_stop = false) {
  s_last_io.stage = stage;
  s_last_io.wire_rc = wire_rc;
  s_last_io.expected = expected;
  s_last_io.received = received;
  s_last_io.used_stop = used_stop;
}

static bool mpu_write8(uint8_t reg, uint8_t val) {
  Wire.beginTransmission(IMU_MPU_ADDR);
  Wire.write(reg);
  Wire.write(val);
  const uint8_t rc = Wire.endTransmission();
  if (rc != 0) {
    set_io_error(MPU_IO_ADDR_WRITE, rc, 0, 0, true);
    return false;
  }
  set_io_error();
  return true;
}

static bool mpu_read_once(uint8_t reg, uint8_t* buf, uint8_t n, bool use_stop) {
  Wire.beginTransmission(IMU_MPU_ADDR);
  Wire.write(reg);
  const uint8_t rc = Wire.endTransmission(use_stop);
  if (rc != 0) {
    set_io_error(MPU_IO_ADDR_WRITE, rc, n, 0, use_stop);
    return false;
  }
  const uint8_t got = (uint8_t)Wire.requestFrom((int)IMU_MPU_ADDR, (int)n);
  if (got != n) {
    while (Wire.available()) (void)Wire.read();
    set_io_error(MPU_IO_DATA_READ, 0, n, got, use_stop);
    return false;
  }
  for (uint8_t i = 0; i < n; i++) buf[i] = Wire.read();
  set_io_error();
  return true;
}

static bool mpu_read(uint8_t reg, uint8_t* buf, uint8_t n) {
  // Repeated-start is the canonical register read. A STOP-separated retry also
  // handles marginal GY-521 boards and mirrors the proven Pololu trunk clients.
  if (mpu_read_once(reg, buf, n, false)) return true;
  delayMicroseconds(50);
  if (mpu_read_once(reg, buf, n, true)) {
    if (!s_stop_fallback_logged) {
      s_stop_fallback_logged = true;
      emit_log("warn", "imu: repeated-start read failed; STOP-separated fallback works");
    }
    return true;
  }
  return false;
}

static const char* io_stage_name(MpuIoStage stage) {
  if (stage == MPU_IO_ADDR_WRITE) return "register-write";
  if (stage == MPU_IO_DATA_READ) return "data-read";
  return "none";
}

static void log_probe_failure(const char* prefix, int observed_who = -1) {
  char msg[176];
  if (observed_who >= 0) {
    snprintf(msg, sizeof(msg), "%s: invalid WHO_AM_I=0x%02X (expected 0x68/0x72/0x98)",
             prefix, observed_who & 0xFF);
  } else {
    snprintf(msg, sizeof(msg),
             "%s: stage=%s wire_rc=%u bytes=%u/%u mode=%s",
             prefix, io_stage_name(s_last_io.stage), (unsigned)s_last_io.wire_rc,
             (unsigned)s_last_io.received, (unsigned)s_last_io.expected,
             s_last_io.used_stop ? "stop" : "repeated-start");
  }
  emit_log("warn", msg);
}

static bool known_who(uint8_t who) {
  return who == 0x68 || who == 0x72 || who == 0x98;
}

static bool probe_mpu(uint8_t& who, int attempts = 4) {
  int invalid_who = -1;
  for (int i = 0; i < attempts; ++i) {
    uint8_t candidate = 0;
    if (mpu_read(0x75, &candidate, 1)) {
      if (known_who(candidate)) {
        who = candidate;
        return true;
      }
      invalid_who = candidate;
    }
    delay(10);
  }
  log_probe_failure("imu: probe failed", invalid_who);
  return false;
}

static bool configure_mpu() {
  if (!mpu_write8(0x6B, 0x80)) return false;
  delay(100);
  return mpu_write8(0x6B, 0x01) &&
         mpu_write8(0x1A, 0x03) &&
         mpu_write8(0x1B, 0x00) &&
         mpu_write8(0x1C, 0x00);
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
  uint8_t who = 0;
  if (!probe_mpu(who)) {
    // The ToF array has already performed substantial bus traffic. Recover the
    // controller once, deselect the mux, and make one clean probe before giving
    // up for now. The runtime tick continues bounded reprobes after boot.
    i2c_trunk_recover("MPU boot probe");
    if (!probe_mpu(who)) {
      s_next_reprobe_ms = millis() + IMU_REPROBE_INTERVAL_MS;
      emit_log("warn", "imu: unavailable at boot; scheduled for periodic reprobe");
      return;
    }
  }
  // Genuine parts report 0x68; known compatible clones report 0x72/0x98.
  char buf[72];
  if (who != 0x68) {
    snprintf(buf, sizeof(buf), "imu: recognized compatible WHO_AM_I=0x%02X", who);
    emit_log("warn", buf);
  }

  // Reset, wake with PLL-X clock, configure filters + full-scale ranges.
  if (!configure_mpu()) {
    log_probe_failure("imu: configuration failed");
    s_next_reprobe_ms = millis() + IMU_REPROBE_INTERVAL_MS;
    return;
  }
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
    log_probe_failure("imu: bias calibration failed");
    s_next_reprobe_ms = millis() + IMU_REPROBE_INTERVAL_MS;
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
  s_ever_online = true;
  s_err_streak = 0;
  s_reprobe_failures = 0;

  snprintf(buf, sizeof(buf), "imu: MPU-6050 online (gyro bias z %+.2f dps, spread %.2f)",
           s_bias_gz, gmax - gmin);
  emit_log("info", buf);
}

bool imu_present() { return s_present; }

static bool runtime_reprobe() {
  uint8_t who = 0;
  if (!probe_mpu(who, 2)) return false;
  if (!configure_mpu()) {
    log_probe_failure("imu: reprobe configuration failed");
    return false;
  }
  delay(50);

  float ax, ay, az, gx, gy, gz;
  if (!mpu_sample(ax, ay, az, gx, gy, gz)) {
    log_probe_failure("imu: reprobe sample failed");
    return false;
  }

  // A runtime recovery must not block the 50 Hz safety task for a one-second
  // calibration. Seed a conservative one-sample bias if this device was never
  // online; otherwise retain the boot-calibrated bias across the outage.
  if (!s_ever_online) {
    s_bias_gx = gx;
    s_bias_gy = gy;
    s_bias_gz = gz;
  }
  s_pitch = accel_pitch(ax, ay, az);
  s_roll = accel_roll(ax, ay, az);
  s_yaw = 0.0f;
  s_present = true;
  s_ever_online = true;
  s_err_streak = 0;
  s_reprobe_failures = 0;
  LOCK_STATE();
  g_ctx.imu.ok = true;
  g_ctx.imu.pitch = s_pitch;
  g_ctx.imu.roll = s_roll;
  g_ctx.imu.yaw = s_yaw;
  UNLOCK_STATE();
  emit_log("info", "imu: MPU-6050 recovered and back online");
  return true;
}

void imu_tick(float dt) {
  if (dt <= 0.0f) return;
  if (!s_present) {
    const uint32_t now = millis();
    if ((int32_t)(now - s_next_reprobe_ms) < 0) return;
    s_next_reprobe_ms = now + IMU_REPROBE_INTERVAL_MS;
    if (++s_reprobe_failures >= IMU_REPROBE_RECOVER_EVERY) {
      s_reprobe_failures = 0;
      i2c_trunk_recover("MPU reprobe backoff");
    }
    (void)runtime_reprobe();
    return;
  }

  float ax, ay, az, gx, gy, gz;
  if (!mpu_sample(ax, ay, az, gx, gy, gz)) {
    // Transient bus error: keep the last attitude. A SOLID failure streak means
    // the sensor is gone/degraded mid-session — recover once, then back off so
    // repeated timeouts do not disturb the rest of the shared I2C trunk.
    if (s_err_streak < 255) s_err_streak++;
    if (s_err_streak == IMU_ERR_STREAK_RECOVER) {
      // Try one controlled controller+line recovery before declaring the MPU
      // temporarily offline. Healthy ToF clients select their mux channel anew.
      i2c_trunk_recover("MPU sustained read errors");
      float rax, ray, raz, rgx, rgy, rgz;
      if (mpu_sample(rax, ray, raz, rgx, rgy, rgz)) {
        s_err_streak = 0;
        emit_log("warn", "imu: sample resumed after I2C trunk recovery");
        return;
      }
      log_probe_failure("imu: offline after recovery");
      s_present = false;
      s_next_reprobe_ms = millis() + IMU_REPROBE_INTERVAL_MS;
      LOCK_STATE();
      g_ctx.imu.ok = false;
      UNLOCK_STATE();
      emit_log("warn", "imu: temporarily offline; periodic reprobe armed");
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
