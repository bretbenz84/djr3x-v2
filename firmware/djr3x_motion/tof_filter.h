// tof_filter.h — shared per-channel distance filter for every ToF publisher
// (radial tof.cpp + matrix tof_matrix.cpp): fast-attack / slow-release, with
// GLITCH CONFIRMATION on large drops.
//
// Field data 2026-07-21 (stationary robot, empty room): channels staring into
// open space sporadically hallucinate a single-frame near return — VL53L1X
// wrap-around phantoms and VL53L7CX speckle/multipath — and the pure
// fast-attack filter believed each one instantly, then took ~1 s to release
// (fl/fr/rr sawtoothing 4 m -> 1 m -> 4 m forever). A real obstacle returns in
// consecutive frames; speckle almost never does. So: a reading that drops the
// state by more than TOF_ATTACK_DROP_MM must repeat (within
// TOF_ATTACK_CONFIRM_MM) on the NEXT frame before it is believed — one revisit
// (~75-80 ms) of extra latency, ~20 mm of travel at the 0.25 m/s cap. Small
// drops (an approaching target moves < DROP_MM per frame) still attack
// instantly, and release stays bounded at TOF_RELEASE_STEP_MM per revisit.
//
// Requires calib.h (the TOF_* knobs) to be included first.
#pragma once
#include <stdint.h>
#include <math.h>

struct TofFilt {
  float state = -1.0f;   // published distance (mm); -1 = unseeded
  float pend  = -1.0f;   // unconfirmed big-drop candidate awaiting a 2nd frame
};

static inline void tof_filter_reset(TofFilt& f) { f.state = -1.0f; f.pend = -1.0f; }

// Feed one VALID reading (mm >= 0); returns the filtered distance to publish.
// Error frames must NOT be fed here — hold/streak policy stays with the caller.
static inline int16_t tof_filter_step(TofFilt& f, int16_t mm) {
  if (f.state < 0.0f) {                        // first valid reading seeds directly
    f.state = (float)mm;
    f.pend = -1.0f;
  } else if ((float)mm <= f.state - (float)TOF_ATTACK_DROP_MM) {
    // Suspiciously large drop: believe it only when two consecutive frames agree.
    if (f.pend >= 0.0f && fabsf((float)mm - f.pend) <= (float)TOF_ATTACK_CONFIRM_MM) {
      f.state = (float)mm;
      f.pend = -1.0f;
    } else {
      f.pend = (float)mm;                      // hold the published state this frame
    }
  } else {
    f.pend = -1.0f;
    if ((float)mm <= f.state) {
      f.state = (float)mm;                     // small/medium drop: attack instantly
    } else {
      const float rise = (float)mm - f.state;  // release toward farther, bounded
      f.state += (rise < (float)TOF_RELEASE_STEP_MM) ? rise : (float)TOF_RELEASE_STEP_MM;
    }
  }
  return (int16_t)(f.state + 0.5f);
}
