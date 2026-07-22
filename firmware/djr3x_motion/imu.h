// imu.h — LSM6DS3 6-axis IMU breakout on the shared I2C trunk (addr 0x6A/0x6B
// by SA0 strap; the probe tries both).
//
// Roadmap Phase A (docs/motion_sensing_roadmap.md §4): the gyro is heading truth
// independent of wheel slip; the accel gives tilt / bump / pickup sensing. This
// module is the FOUNDATION layer: probe, configure, calibrate gyro bias at boot
// (the base boots idle, so rest-time calibration is natural), and publish a
// complementary-filtered attitude (pitch/roll vs gravity + integrated yaw) into
// g_ctx.imu for telemetry. Fusion INTO odometry/turn control is a later phase —
// nothing in the control loop consumes this yet.
//
// WIRING: VCC->3V3, GND->GND, SDA->GPIO21, SCL->GPIO22 (piggybacks the ToF/INA
// trunk; 0x6A/0x6B collides with nothing — mux 0x70, sensors behind mux at
// 0x29, INA226 0x40, QMC5883P 0x2C). Mount near the axle midpoint, away from motor cables.
//
// THREADING: all reads happen on the sensor task (the only I2C task), like ToF
// and the INA226 — never from control/serial. Publishes under the state lock.
//
// No sensor wired? imu_init() validates WHO_AM_I, attempts one controlled trunk
// recovery, then leaves g_ctx.imu.ok false and reprobes on a bounded backoff.
#pragma once

// Call once from setup() AFTER tof/battery init (the central trunk is already up).
void imu_init();

// Call every sensor-task tick (50 Hz). dt = seconds since the previous call.
// Reads accel+gyro, runs the complementary filter, publishes g_ctx.imu.
void imu_tick(float dt);

// True while an LSM6DS3 is online (including after a successful runtime reprobe).
bool imu_present();
