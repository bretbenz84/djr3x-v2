// Front-right radial sensor: time-confirm discontinuities before publishing.
// The other radial channels and 8x8 matrix keep their existing fast filter.
#pragma once
#include <stdint.h>
#include <stdlib.h>
#include "calib.h"

static inline int16_t tof_fr_input_mm(int mm, bool valid, bool out_of_range) {
  if (valid && mm >= 0) return mm < TOF_L1X_OUT_OF_RANGE_MM ? mm : TOF_L1X_OUT_OF_RANGE_MM;
  return out_of_range ? TOF_L1X_OUT_OF_RANGE_MM : -1;
}

// Quality hysteresis, independent of distance. A phase-failing channel must
// not be resurrected by occasional valid-looking near returns. -1 is unavailable,
// never a synthetic clear range. Recovery requires fresh, uninterrupted evidence.
struct TofFrHealth {
  uint16_t failures = 0;
  uint8_t count = 0, good_run = 0;
  uint32_t last_ms = 0;
  bool quarantined = false;
};

static inline bool tof_fr_health_step(TofFrHealth& h, bool valid, uint32_t now_ms) {
  if (h.count && now_ms == h.last_ms) return !h.quarantined;
  if (h.count && (uint32_t)(now_ms - h.last_ms) > TOF_FR_SAMPLE_GAP_MS) {
    h.quarantined = true;
    h.good_run = 0;
  }
  h.last_ms = now_ms;
  h.failures = (uint16_t)((h.failures << 1) | (valid ? 0 : 1));
  if (h.count < 16) ++h.count;
  h.good_run = valid ? (h.good_run < 16 ? h.good_run + 1 : 16) : 0;
  unsigned errors = 0;
  for (uint16_t bits = h.failures; bits; bits >>= 1) errors += bits & 1;
  if (errors >= 8) h.quarantined = true;
  if (h.quarantined && h.good_run >= 16) {
    h.quarantined = false;
    h.failures = 0;
  }
  return !h.quarantined;
}

struct TofFrFilter {
  int16_t state = -1;
  int16_t candidate = -1;
  int8_t direction = 0;          // -1 nearer, +1 farther
  uint8_t samples = 0;
  uint32_t since_ms = 0, last_ms = 0;
};

static inline void tof_fr_cancel_candidate(TofFrFilter& f) {
  f.candidate = -1; f.direction = 0; f.samples = 0;
}

static inline int16_t tof_fr_filter_step(TofFrFilter& f, int16_t mm, uint32_t now_ms) {
  if (mm < 0) {                 // an invalid read proves neither near nor clear
    tof_fr_cancel_candidate(f);
    return f.state;             // caller's bounded error streak owns staleness
  }
  if (f.state < 0) {            // cold start: conservatively honor the first range
    f.state = mm;
    tof_fr_cancel_candidate(f);
    return f.state;
  }
  const int drop = (int)f.state - mm;
  const bool near_jump = drop >= TOF_ATTACK_DROP_MM ||
      (mm < TOF_FR_NEAR_MM && drop >= TOF_FR_NEAR_DROP_MM);
  const bool far_jump = -drop >= TOF_ATTACK_DROP_MM;
  const int8_t direction = near_jump ? -1 : (far_jump ? 1 : 0);
  if (direction) {
    if (f.direction != direction || !f.samples ||
        (uint32_t)(now_ms - f.last_ms) > TOF_FR_SAMPLE_GAP_MS ||
        abs((int)mm - f.candidate) > TOF_FR_CONFIRM_TOLERANCE_MM) {
      f.direction = direction; f.candidate = mm;
      f.since_ms = f.last_ms = now_ms; f.samples = 1;
      return f.state;
    }
    // Repeated calls for a cached observation cannot establish persistence.
    if (now_ms == f.last_ms) return f.state;
    f.last_ms = now_ms;
    if (f.samples < 255) ++f.samples;
    const uint32_t hold_ms = near_jump ? TOF_FR_NEAR_CONFIRM_MS : TOF_FR_CLEAR_CONFIRM_MS;
    if (f.samples < 3 || (uint32_t)(now_ms - f.since_ms) < hold_ms) return f.state;
    if (near_jump) {
      f.state = mm;             // persistent obstruction: accept, then track normally
      tof_fr_cancel_candidate(f);
      return f.state;
    }
    // A confirmed clear run may keep releasing; do not re-arm every 300 mm.
  } else {
    tof_fr_cancel_candidate(f);
  }
  if (mm <= f.state) f.state = mm;   // smooth approach: no extra detection delay
  else {
    const int rise = (int)mm - f.state;
    f.state += rise < TOF_RELEASE_STEP_MM ? rise : TOF_RELEASE_STEP_MM;
  }
  return f.state;
}
