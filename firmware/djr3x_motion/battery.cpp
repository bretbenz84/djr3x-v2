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
#include <Preferences.h>   // ESP32 NVS — the SOC ledger survives USB power-off
#include "battery.h"
#include "context.h"
#include "pins.h"
#include "proto_io.h"      // emit_log — gauge anchor/restore diagnostics

#ifndef BATT_INA226_ADDR
#define BATT_INA226_ADDR 0x40      // A0/A1 to GND (module default)
#endif
#ifndef BATT_SHUNT_MICROOHM
#define BATT_SHUNT_MICROOHM 0      // 0 = voltage-only (stock 100 mOhm module
                                   // shunt maxes at +/-0.8A — not motor-ranged)
#endif
#ifndef BATT_CURRENT_SIGN
#define BATT_CURRENT_SIGN 1        // +1 if IN+ faces the battery; -1 if it faces
                                   // the load (calib.h sets the as-built value)
#endif

static bool  s_present = false;
static float s_mv_ema  = -1.0f;
static float s_ma_ema  = 0.0f;

// ---- SOC gauge state (see calib.h "Battery gauge") ----
static Preferences s_prefs;               // NVS namespace "batt", key "mah"
static float    s_soc_mah       = -1.0f;  // remaining mAh; -1 = not initialized yet
static float    s_saved_mah     = -1.0f;  // last value persisted to NVS
static uint32_t s_saved_at_ms   = 0;
static int      s_quiet_ticks   = 0;      // consecutive 1 Hz ticks at rest
static bool     s_full_anchored = false;  // full-anchor fired since last discharge dip
static uint32_t s_last_tick_ms  = 0;      // for the Ah integration dt
static volatile bool s_mark_full_req = false;  // "batt_full" flag: set on the serial
                                               // task, consumed by battery_tick

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

#if BATT_SHUNT_MICROOHM > 0
  // Restore the SOC ledger from NVS. Boot-time voltage reconciliation (full
  // anchor / knee clamps / plateau fallback) happens in battery_tick once the
  // first valid quiet readings arrive — VBUS needs a moment to be sampled.
  s_prefs.begin("batt", false);
  s_soc_mah = s_prefs.getFloat("mah", -1.0f);
  s_saved_mah = s_soc_mah;
  if (s_soc_mah >= 0.0f) {
    char buf[64];
    snprintf(buf, sizeof(buf), "battery: SOC ledger restored (%.0f mAh, %.0f%%)",
             s_soc_mah, 100.0f * s_soc_mah / (float)BATT_CAPACITY_MAH);
    emit_log("info", buf);
  }
#endif
}

#if BATT_SHUNT_MICROOHM > 0
// Persist the ledger when it moved enough or enough time passed (NVS wear-safe:
// worst case ~6 writes/hour; the partition wear-levels far beyond that).
static void soc_maybe_save(uint32_t now) {
  if (s_soc_mah < 0.0f) return;
  const bool moved = fabsf(s_soc_mah - s_saved_mah) >= (float)BATT_SOC_SAVE_DELTA_MAH;
  const bool due   = (uint32_t)(now - s_saved_at_ms) >= (uint32_t)BATT_SOC_SAVE_SECS * 1000u;
  if (!moved && !due) return;
  s_prefs.putFloat("mah", s_soc_mah);
  s_saved_mah = s_soc_mah;
  s_saved_at_ms = now;
}

// Coarse plateau estimate for a first boot with no ledger (rest voltage, 4S LiFePO4).
static float soc_from_rest_mv(float mv) {
  if (mv >= BATT_SOC_FULL_ANCHOR_MV) return 1.00f;
  if (mv >= 13150.0f) return 0.70f;
  if (mv >= 13000.0f) return 0.40f;
  if (mv >= (float)BATT_SOC_KNEE1_MV) return 0.25f;
  if (mv >= (float)BATT_SOC_KNEE2_MV) return 0.12f;
  return 0.05f;
}
#endif

bool battery_present() { return s_present; }

bool battery_gauge_available() {
#if BATT_SHUNT_MICROOHM > 0
  return s_present;
#else
  return false;
#endif
}

void battery_request_mark_full() { s_mark_full_req = true; }

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
    // 2.5 uV/LSB across the shunt; I = V/R. Signed register; BATT_CURRENT_SIGN
    // maps the as-built sense orientation onto "+ = discharging" (§6.1).
    float uv = (int16_t)sraw * 2.5f;
    float ma = (float)BATT_CURRENT_SIGN * uv * 1000.0f / (float)BATT_SHUNT_MICROOHM;
    s_ma_ema = 0.8f * s_ma_ema + 0.2f * ma;
  }
#endif

#if BATT_SHUNT_MICROOHM > 0
  // ---- SOC gauge (1 Hz tick) ----
  const uint32_t now = millis();
  const float dt_h = (s_last_tick_ms == 0) ? 0.0f
                     : (float)(uint32_t)(now - s_last_tick_ms) / 3600000.0f;
  s_last_tick_ms = now;

  // Rest tracking: no motor drive (idle electronics ~1 A = C/40, negligible sag).
  const bool quiet = fabsf(s_ma_ema) < (float)BATT_SOC_QUIET_MA;
  s_quiet_ticks = quiet ? s_quiet_ticks + 1 : 0;

  // Host-commanded full mark (batt_full): the operator watched the charger's
  // taper current hit cutoff, which is BETTER evidence than our rest-voltage
  // anchor (it works mid-absorption, when current still flows and the pack
  // never looks "quiet"). Outranks the ledger like the boot anchor; persisted
  // immediately so a power-off right after the click can't lose it. Also
  // initializes a ledger that never existed (s_soc_mah == -1).
  if (s_mark_full_req) {
    s_mark_full_req = false;
    s_soc_mah = (float)BATT_CAPACITY_MAH;
    s_full_anchored = true;   // same once-per-charge arming as the rest anchor
    s_prefs.putFloat("mah", s_soc_mah);
    s_saved_mah = s_soc_mah;
    s_saved_at_ms = now;
    emit_log("info", "battery: host marked pack full - SOC set to 100%");
  }

  if (s_soc_mah < 0.0f && have_mv && s_quiet_ticks >= BATT_SOC_ANCHOR_TICKS) {
    // First boot ever (no ledger): coarse init from the rest voltage.
    s_soc_mah = soc_from_rest_mv(s_mv_ema) * (float)BATT_CAPACITY_MAH;
    emit_log("info", "battery: SOC initialized from rest voltage (no ledger)");
  }

  if (s_soc_mah >= 0.0f) {
    // Coulomb count: + ma = discharging (sign handled above). Charging through
    // the shunt (if ever wired that way) counts back in for free.
    s_soc_mah -= s_ma_ema * dt_h;
    s_soc_mah = clampf(s_soc_mah, 0.0f, (float)BATT_CAPACITY_MAH);

    if (have_mv && s_quiet_ticks >= BATT_SOC_ANCHOR_TICKS) {
      // FULL anchor: rest voltage at/above the anchor = the pack was charged
      // while we were dark -> 100%. Once per charge (rearms after a real dip).
      if (s_mv_ema >= (float)BATT_SOC_FULL_ANCHOR_MV && !s_full_anchored) {
        s_soc_mah = (float)BATT_CAPACITY_MAH;
        s_full_anchored = true;
        emit_log("info", "battery: rest voltage at full anchor - SOC reset to 100%");
      } else if (s_mv_ema < (float)BATT_SOC_FULL_ANCHOR_MV - 100.0f) {
        s_full_anchored = false;
      }
      // KNEE clamps: the sharp end of the LiFePO4 curve outranks the ledger —
      // clamp DOWN only (never up: a sagging ledger must not be inflated).
      const float knee1 = (float)BATT_CAPACITY_MAH * BATT_SOC_KNEE1_PCT / 100.0f;
      const float knee2 = (float)BATT_CAPACITY_MAH * BATT_SOC_KNEE2_PCT / 100.0f;
      if (s_mv_ema < (float)BATT_SOC_KNEE2_MV && s_soc_mah > knee2) {
        s_soc_mah = knee2;
        emit_log("warn", "battery: rest voltage below low knee - SOC clamped (pack is LOW)");
      } else if (s_mv_ema < (float)BATT_SOC_KNEE1_MV && s_soc_mah > knee1) {
        s_soc_mah = knee1;
        emit_log("info", "battery: rest voltage below knee - SOC clamped");
      }
    }
    soc_maybe_save(now);
  }
#endif

  LOCK_STATE();
  if (have_mv) g_ctx.batt_mv = (int16_t)(s_mv_ema + 0.5f);
#if BATT_SHUNT_MICROOHM > 0
  g_ctx.batt_ma = (int16_t)s_ma_ema;
  g_ctx.batt_soc = (s_soc_mah >= 0.0f)
      ? (int8_t)(100.0f * s_soc_mah / (float)BATT_CAPACITY_MAH + 0.5f) : (int8_t)-1;
#endif
  UNLOCK_STATE();
}
