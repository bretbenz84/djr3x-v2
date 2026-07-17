// env.h — Bosch BMP280/BME280 environment sensor (temp / pressure / humidity*).
//
// A GY-BME280-style breakout on the main I2C trunk (GPIO21/22) gives Rex a sense
// of the room: air temperature, barometric pressure, and — ONLY if the fitted
// chip is a genuine BME280 (many "BME280" listings ship BMP280s; chip ID tells
// the truth at probe time) — relative humidity. Polled slowly (an office climate
// does not change at 50 Hz); values ride telemetry in the `env` block.
//
// env_init(): probe 0x76 then 0x77, read chip ID (0x58 = BMP280, 0x60 = BME280),
//             load factory calibration, configure continuous low-rate sampling.
// env_tick(): one compensated read → g_ctx.env. Call at ~0.5 Hz or slower.
#pragma once

void env_init();
void env_tick();
bool env_present();
