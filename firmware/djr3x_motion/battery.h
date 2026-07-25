// battery.h — pack voltage (and optional current) via an INA226 on the shared I2C bus.
//
// WIRING (solder-and-flash; see README "Battery sense"):
//   INA226 breakout: VCC->3V3, GND->GND, SDA->GPIO21, SCL->GPIO22 (piggyback the
//   ToF bus — the INA's 0x40 address does not collide with the ToF mux at 0x70),
//   VBUS->BATT+ (the 12.8V LiFePO4 positive terminal). That's it for VOLTAGE.
//   CURRENT (optional, later): the stock module shunt (R100 = 100 mOhm) only
//   ranges +/-0.8A — useless for drive motors. Fit a 2 mOhm shunt inline in the
//   main battery lead and set BATT_SHUNT_MICROOHM accordingly; until then leave
//   it 0 and only bus voltage is reported.
//
// No sensor wired? battery_init() probes the address once; absent -> batt_mv
// stays -1 ("unknown") and the host feature stays dormant. The old firmware
// stubbed batt_mv=12000, which the host could not distinguish from a real 12.0V
// pack — -1 is the explicit unknown.

#pragma once
#include <stdint.h>

// Call once from setup() after tof_init() (shares Wire).
void battery_init();

// Call ~1 Hz from sensorTask. Updates g_ctx.batt_mv (and g_ctx.batt_ma when a
// real shunt is configured) under the state lock.
void battery_tick();

// True when an INA226 answered the probe at boot.
bool battery_present();

// True when this build carries the coulomb SOC gauge AND the sensor is alive
// (BATT_SHUNT_MICROOHM > 0 and the INA226 answered the boot probe).
bool battery_gauge_available();

// Host command "batt_full" (docs §5.11 — the Mac menu bar meter clicks this when
// the charger's taper current says the pack is done): request the SOC ledger be
// set to 100%. Thread-safe from the serial task: only sets a flag; the next 1 Hz
// battery_tick on sensorTask applies it and persists to NVS immediately.
void battery_request_mark_full();

// Host command "batt_soc": set the SOC ledger to an ARBITRARY percentage. Same
// thread-safety and persistence as battery_request_mark_full, but for the case
// batt_full cannot express — correcting the gauge after a mis-click, or seeding it
// from a known state (owner 2026-07-24: an accidental "Set Battery to 100%" while
// reaching for the joystick left a 67% pack reading full, with no way to say 67).
// pct is clamped to 0..100. Values below 100 also clear the full-anchor latch so the
// rest-voltage anchor can legitimately re-arm.
void battery_request_set_soc(float pct);
