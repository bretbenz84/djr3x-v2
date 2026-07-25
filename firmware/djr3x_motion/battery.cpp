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
static volatile bool  s_set_soc_req = false;   // "batt_soc" flag (same handoff)
static volatile float s_set_soc_pct = -1.0f;   // 0..100, valid while s_set_soc_req
static bool s_charging = false;    // debounced on-charger state (calib.h knobs)
static int  s_chg_ticks = 0;       // consecutive ticks toward the pending edge

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

void battery_request_set_soc(float pct) {
  if (pct < 0.0f) pct = 0.0f;
  if (pct > 100.0f) pct = 100.0f;
  s_set_soc_pct = pct;      // write the value BEFORE arming, so battery_tick can
  s_set_soc_req = true;     // never observe the flag with a stale percentage
}

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

  // Rest tracking: idle-electronics draw only. Any INFLOW disqualifies rest
  // outright — the charger DOES cross the shunt in the current wiring (the old
  // "charge current never crosses the shunt" note is stale), and during its
  // ramp/taper the current dips under the quiet bar while the terminals sit at
  // SUPPLY voltage: 20 such ticks anchored a 22% pack to "100%" (field
  // 2026-07-17, plug-in event 00:31). Charging is handled by coulomb counting
  // + the taper-time batt_full mark, never by voltage anchors.
  const bool inflow = (s_ma_ema < -(float)BATT_CHARGE_DETECT_MA) || s_charging;
  const bool quiet = !inflow && fabsf(s_ma_ema) < (float)BATT_SOC_QUIET_MA;
  s_quiet_ticks = quiet ? s_quiet_ticks + 1 : 0;

  // Rest-voltage ESTIMATE for the anchors: back out the IR sag of the idle draw
  // (~160 mΩ junction: even 1.3 A hides 0.2 V). Only meaningful when quiet.
  const float rest_mv = s_mv_ema + s_ma_ema * (float)BATT_PACK_IR_MOHM / 1000.0f;

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

  // Host-commanded arbitrary SOC (batt_soc): the operator KNOWS the real level —
  // e.g. correcting an accidental mark-full. Outranks the ledger exactly like the
  // full mark, and is persisted immediately for the same reason.
  if (s_set_soc_req) {
    s_set_soc_req = false;
    const float pct = s_set_soc_pct;
    s_soc_mah = (pct / 100.0f) * (float)BATT_CAPACITY_MAH;
    // Only a genuine 100% should leave the full anchor armed; anything less must
    // re-arm it so a later real charge can still anchor at the top.
    s_full_anchored = (pct >= 100.0f);
    s_prefs.putFloat("mah", s_soc_mah);
    s_saved_mah = s_soc_mah;
    s_saved_at_ms = now;
    char buf[64];
    snprintf(buf, sizeof(buf), "battery: host set SOC to %.0f%%", pct);
    emit_log("info", buf);
  }

  if (s_soc_mah < 0.0f && have_mv && s_quiet_ticks >= BATT_SOC_ANCHOR_TICKS) {
    // First boot ever (no ledger): coarse init from the rest voltage.
    s_soc_mah = soc_from_rest_mv(rest_mv) * (float)BATT_CAPACITY_MAH;
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
      if (rest_mv >= (float)BATT_SOC_FULL_ANCHOR_MV && !s_full_anchored) {
        s_soc_mah = (float)BATT_CAPACITY_MAH;
        s_full_anchored = true;
        emit_log("info", "battery: rest voltage at full anchor - SOC reset to 100%");
      } else if (rest_mv < (float)BATT_SOC_FULL_ANCHOR_MV - 100.0f) {
        s_full_anchored = false;
      }
      // KNEE clamps: the sharp end of the LiFePO4 curve outranks the ledger —
      // clamp DOWN only (never up: a sagging ledger must not be inflated).
      const float knee1 = (float)BATT_CAPACITY_MAH * BATT_SOC_KNEE1_PCT / 100.0f;
      const float knee2 = (float)BATT_CAPACITY_MAH * BATT_SOC_KNEE2_PCT / 100.0f;
      if (rest_mv < (float)BATT_SOC_KNEE2_MV && s_soc_mah > knee2) {
        s_soc_mah = knee2;
        emit_log("warn", "battery: rest voltage below low knee - SOC clamped (pack is LOW)");
      } else if (rest_mv < (float)BATT_SOC_KNEE1_MV && s_soc_mah > knee1) {
        s_soc_mah = knee1;
        emit_log("info", "battery: rest voltage below knee - SOC clamped");
      }
    }
    soc_maybe_save(now);
  }
#endif

#if BATT_SHUNT_MICROOHM > 0
  // ---- Charging detection (debounced both ways; see calib.h) ----
  // Enter on definite current flowing into the pack. Once latched, do NOT release
  // merely because a full charger's taper/cutoff falls near 0 mA — that was letting
  // the wheels wake up while the cable was still attached. Only sustained current
  // flowing OUT of the pack proves the charger is no longer carrying the load.
  // Hysteresis: ENTER on the high threshold (clearly the charger), but once latched
  // HOLD down to the lower EXIT floor, so a servo-load voltage sag can't false-release
  // the drive lockout (field 2026-07-23). See calib.h BATT_CHARGE_EXIT_MV.
  const bool charger_voltage_enter = have_mv && s_mv_ema >= (float)BATT_CHARGE_DETECT_MV;
  const bool charger_voltage_hold  = have_mv && s_mv_ema >= (float)BATT_CHARGE_EXIT_MV;
  const bool chg_now = s_charging
      ? (charger_voltage_hold || s_ma_ema < (float)BATT_CHARGE_EXIT_DISCHARGE_MA)
      : (charger_voltage_enter || s_ma_ema <= -(float)BATT_CHARGE_DETECT_MA);
  if (chg_now != s_charging) {
    s_chg_ticks++;
    const int need = s_charging ? BATT_CHARGE_EXIT_TICKS : BATT_CHARGE_ENTER_TICKS;
    if (s_chg_ticks >= need) {
      s_charging = chg_now;
      s_chg_ticks = 0;
      emit_event_kv("charging", "state", s_charging ? "on" : "off");
      emit_log("info", s_charging
               ? "battery: charger detected - drive locked out"
               : "battery: charger disconnected - drive released");
    }
  } else {
    s_chg_ticks = 0;
  }
#endif

  LOCK_STATE();
  if (have_mv) g_ctx.batt_mv = (int16_t)(s_mv_ema + 0.5f);
#if BATT_SHUNT_MICROOHM > 0
  g_ctx.batt_ma = (int16_t)s_ma_ema;
  g_ctx.charging = s_charging;
  g_ctx.batt_soc = (s_soc_mah >= 0.0f)
      ? (int8_t)(100.0f * s_soc_mah / (float)BATT_CAPACITY_MAH + 0.5f) : (int8_t)-1;
#endif
  UNLOCK_STATE();
}
