// fusion.cpp — rotation, confidence falloff, seam dedup (pure; see fusion.h).
#include "fusion.h"
#include "calib.h"
#include <math.h>
#include <string.h>

static const float DEG2RAD = 0.017453292519943295f;
static const float RAD2DEG = 57.29577951308232f;

float radar_wrap180(float deg) {
  float d = fmodf(deg + 180.0f, 360.0f);
  if (d < 0.0f) d += 360.0f;
  return d - 180.0f + (d == 0.0f ? 360.0f : 0.0f);   // (-180, 180]: -180 -> +180
}

// Edge-confidence falloff (calib.h): 1.0 inside ±FULL, linear to EDGE_MIN at
// ±EDGE, BEYOND_EDGE past the rated FOV.
static float edge_confidence(float local_deg) {
  const float a = fabsf(local_deg);
  if (a <= RADAR_CONF_FULL_DEG) return 1.0f;
  if (a >= RADAR_CONF_EDGE_DEG) return RADAR_CONF_BEYOND_EDGE;
  const float f = (a - RADAR_CONF_FULL_DEG) / (RADAR_CONF_EDGE_DEG - RADAR_CONF_FULL_DEG);
  return 1.0f + f * (RADAR_CONF_EDGE_MIN - 1.0f);
}

bool radar_local_to_robot(const Ld2450Target* in, int sensor_idx,
                          float mount_deg, bool flip_x, RadarTargetRobot* out) {
  if (!in->present) return false;
  const float x = flip_x ? -(float)in->x_mm : (float)in->x_mm;
  const float y = (float)in->y_mm;
  const float range_m = sqrtf(x * x + y * y) * 0.001f;
  if (range_m < RADAR_RANGE_MIN_M || range_m > RADAR_RANGE_MAX_M) return false;
  // +x = right of sensor (official convention), + bearing = left/CCW, so the
  // lateral axis negates — same expression ESPHome uses: atan2(-x, y).
  const float local_deg = atan2f(-x, y) * RAD2DEG;
  out->bearing_deg = radar_wrap180(mount_deg + local_deg);
  out->range_m     = range_m;
  out->confidence  = edge_confidence(local_deg);
  out->speed_mps   = (float)in->speed_cms * 0.01f;
  out->sensors     = (uint8_t)(1u << sensor_idx);
  return true;
}

static bool mergeable(const RadarTargetRobot* a, const RadarTargetRobot* b) {
  const float db = fabsf(radar_wrap180(a->bearing_deg - b->bearing_deg));
  const float dr = fabsf(a->range_m - b->range_m);
  return db <= RADAR_DEDUP_BEARING_DEG && dr <= RADAR_DEDUP_RANGE_M;
}

int radar_fuse(const RadarTargetRobot* in, int n_in,
               RadarTargetRobot* out, int max_out) {
  if (n_in <= 0 || max_out <= 0) return 0;
  if (n_in > RADAR_MAX_RAW_TARGETS) n_in = RADAR_MAX_RAW_TARGETS;

  bool used[RADAR_MAX_RAW_TARGETS];
  memset(used, 0, sizeof(used));
  int n_out = 0;

  // Greedy, seeded by the most confident unmerged target each round — n is at
  // most sensors*3, so O(n^2) is nothing.
  for (;;) {
    int seed = -1;
    for (int i = 0; i < n_in; i++) {
      if (!used[i] && (seed < 0 || in[i].confidence > in[seed].confidence)) seed = i;
    }
    if (seed < 0 || n_out >= max_out) break;

    // Accumulate the cluster around the seed: confidence-weighted circular
    // mean for bearing (a naive average is wrong across the ±180 wrap).
    float wsum = 0, cx = 0, cy = 0, range = 0, speed = 0;
    float miss = 1.0f;          // product of (1 - c): agreement raises confidence
    uint8_t sensors = 0;
    for (int i = 0; i < n_in; i++) {
      if (used[i] || !mergeable(&in[seed], &in[i])) continue;
      used[i] = true;
      const float w = in[i].confidence > 0.01f ? in[i].confidence : 0.01f;
      wsum  += w;
      cx    += w * cosf(in[i].bearing_deg * DEG2RAD);
      cy    += w * sinf(in[i].bearing_deg * DEG2RAD);
      range += w * in[i].range_m;
      speed += w * in[i].speed_mps;
      miss  *= (1.0f - (in[i].confidence > 1.0f ? 1.0f : in[i].confidence));
      sensors |= in[i].sensors;
    }
    RadarTargetRobot& o = out[n_out++];
    o.bearing_deg = radar_wrap180(atan2f(cy, cx) * RAD2DEG);
    o.range_m     = range / wsum;
    o.speed_mps   = speed / wsum;
    o.confidence  = 1.0f - miss;
    o.sensors     = sensors;
  }

  // Best-first, so the Mac's "start looking here" is out[0].
  for (int i = 1; i < n_out; i++) {
    RadarTargetRobot key = out[i];
    int j = i - 1;
    while (j >= 0 && out[j].confidence < key.confidence) { out[j + 1] = out[j]; j--; }
    out[j + 1] = key;
  }
  return n_out;
}
