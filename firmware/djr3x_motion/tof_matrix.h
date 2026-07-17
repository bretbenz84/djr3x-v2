// tof_matrix.h — DFRobot 8x8 Matrix ToF (SEN0628, VL53L7CX + onboard RP2040) as the
// FRONT obstacle sensor. Mounted at the direct front of the base, pointed level.
//
// Gated by MOTION_TOF_MATRIX_PRESENT (hal.h), independent of the 8-radial
// MOTION_TOF_PRESENT array. The 64-zone depth grid is reduced to TWO numbers —
// the nearest OBSTACLE (floor-rejected, horizontal-projected) in the left and
// right halves of the field of view — which override TofMm.fl / TofMm.fr, so the
// whole existing zone reflex / telemetry / Mac stack works unchanged.
//
// FLOOR REJECTION: the sensor sits TOF_MATRIX_HEIGHT_M above the floor with a
// ~45° vertical FOV, so the lower rows permanently see the floor at short range
// (bottom row ≈ 0.45 m at h=0.15 m) — raw, that would pin the front zone in
// SLOW/STOP forever. Each row below the horizon gets a geometric floor-distance
// expectation (h / sin(row angle)); readings at/beyond ~TOF_MATRIX_FLOOR_TOLERANCE
// of it are floor (clear), readings meaningfully SHORTER are real obstacles
// (chair legs included — and the near-horizontal rows see legs/seats at full
// range regardless). See tof_matrix.cpp for the math + calib.h for the knobs.
#pragma once
#include <stdint.h>

// Bring the sensor up. NON-BLOCKING: spawns a one-shot low-priority init task
// (the module's ranging-mode switch needs a ~5 s settle that must not stall
// setup()/the Mac handshake). Until init completes the read below contributes
// nothing. Safe to call when the sensor is absent (init retries in background,
// logging every few attempts).
void tof_matrix_init();

// True once the sensor is initialized and streaming frames.
bool tof_matrix_ready();

// Latest front obstacle distances (mm, horizontal-projected, floor-rejected):
// *fl = nearest obstacle in the LEFT half of the FOV, *fr = RIGHT half.
// TOF_MATRIX_CLEAR_MM = nothing in range. -1 = sensor error / not ready (the
// caller decides how to combine with the radial array / stub).
// Rate-limited internally (TOF_MATRIX_FRAME_INTERVAL_MS); cheap between frames.
// Call from the sensor task only (single consumer, no locking inside).
void tof_matrix_read(int16_t* fl, int16_t* fr);
