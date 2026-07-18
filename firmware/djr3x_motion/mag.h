// mag.h — QMC5883L magnetometer (GY-271 board) raw-axis publisher.
//
// The compass joins the main I2C trunk (GPIO21/22) at fixed 7-bit address 0x0D
// — clear of the TCA9548A mux (0x70), INA226 (0x40), MPU-6050 (0x68), and the
// BMP/BME280 (0x76). NOTE this is the QMC variant, NOT the HMC5883L: different
// register map entirely (continuous-mode config at 0x09, data at 0x00..0x05
// little-endian, chip-id 0xFF at register 0x0D).
//
// The firmware's job is deliberately minimal: probe, configure continuous mode,
// and publish RAW int16 axes in telemetry (`mag` block) at ~10 Hz. Everything
// smart — hard/soft-iron calibration, tilt compensation against the IMU's
// pitch/roll, current-gated fusion against batt_ma — lives on the Mac
// (hardware/compass.py), where it is unit-testable without hardware and the
// calibration persists as JSON. Raw counts on the wire keep this boundary clean.
//
// mag_init(): probe + configure. Absent sensor logs once and stays {ok:false}.
// mag_tick(): one DRDY-gated read -> g_ctx.mag. Call at ~10 Hz.
#pragma once

void mag_init();
void mag_tick();
bool mag_present();
