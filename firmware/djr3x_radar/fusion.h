// fusion.h — sensor-local targets -> robot-frame bearings, seam dedup,
// edge-confidence falloff.
//
// PURE module (no Arduino includes) — compiles on the HOST for the regression
// harness and on the ESP32-S3 unchanged. All functions are stateless
// snapshot-in/snapshot-out, in the mold of the drive base's mx_aggregate.
//
// Frame convention (docs/motion_protocol.md §4, same as turn.deg/come.heading
// and the Mac's ToF bearing table): degrees, 0 = robot forward, + = LEFT/CCW,
// wrapped to (-180, 180]. The Mac can feed a fused bearing straight into a
// `turn` with no conversion.
#pragma once
#include <stdint.h>
#include "ld2450.h"

// One radar return in the robot frame.
struct RadarTargetRobot {
  float   bearing_deg;  // 0 = forward, + = left/CCW, (-180, 180]
  float   range_m;
  float   confidence;   // 0..1 — falls off toward each sensor's FOV edges;
                        // RAISED when two sensors agree across a seam
  float   speed_mps;    // radial, sign as the module reports it (+ away, per
                        // ESPHome's reading — unofficial, verify on hardware)
  uint8_t sensors;      // bitmask of contributing sensor indices
};

float radar_wrap180(float deg);

// Rotate one sensor-local return into the robot frame and stamp its edge
// confidence. mount_deg is the sensor's boresight bearing (pins.h table);
// flip_x mirrors the sensor's lateral axis (calib.h RADAR_FLIP_X — see the
// x-polarity note in ld2450.h). Returns false when the return is out of the
// plausible range band (RADAR_RANGE_MIN/MAX_M) and should be discarded.
bool radar_local_to_robot(const Ld2450Target* in, int sensor_idx,
                          float mount_deg, bool flip_x, RadarTargetRobot* out);

// Merge robot-frame targets that lie within RADAR_DEDUP_BEARING_DEG AND
// RADAR_DEDUP_RANGE_M of each other (a person near a mount seam is reported by
// both adjacent sensors). Confidence-weighted circular mean; agreement raises
// confidence. Output is sorted by confidence, best first. Returns the count
// written to out (<= max_out).
int radar_fuse(const RadarTargetRobot* in, int n_in,
               RadarTargetRobot* out, int max_out);
