#include "config_params.h"

// Read one float key; if present, clamp to [lo,hi] and report whether clamped.
static bool take_f(JsonObjectConst c, const char* key, float lo, float hi, float& dst) {
  if (!c[key].is<float>() && !c[key].is<int>()) return false;
  float raw = c[key].as<float>();
  float cl  = clampf(raw, lo, hi);
  dst = cl;
  return cl != raw;  // true => was clamped
}
static bool take_u(JsonObjectConst c, const char* key, uint32_t lo, uint32_t hi, uint32_t& dst) {
  if (!c[key].is<unsigned>() && !c[key].is<int>()) return false;
  long raw = c[key].as<long>();
  uint32_t v = raw < 0 ? 0u : (uint32_t)raw;
  uint32_t cl = clampu(v, lo, hi);
  dst = cl;
  return cl != (uint32_t)raw;
}

bool apply_config(JsonObjectConst cmd, MotionParams& out) {
  // Work on a local copy seeded from current params, then commit under lock.
  MotionParams p;
  LOCK_STATE();
  p = g_ctx.params;
  UNLOCK_STATE();

  bool clamped = false;
  // Caps clamp against the compile-time hard ceilings (config can only tighten).
  clamped |= take_f(cmd, "max_lin",        0.0f, HARDCAP_MAX_LINEAR_MS,     p.max_lin);
  clamped |= take_f(cmd, "max_ang",        0.0f, HARDCAP_MAX_ANGULAR_RAD_S, p.max_ang);  // rad/s on the wire
  clamped |= take_f(cmd, "slow_zone_m",    0.0f, 5.0f,                      p.slow_zone_m);
  clamped |= take_f(cmd, "stop_zone_m",    0.0f, 5.0f,                      p.stop_zone_m);
  clamped |= take_f(cmd, "come_stop_at_m", 0.0f, 5.0f,                      p.come_stop_at_m);
  clamped |= take_f(cmd, "default_turn_deg",  0.0f, 360.0f,                 p.default_turn_deg);
  clamped |= take_f(cmd, "default_turn_rate", 0.0f, HARDCAP_MAX_TURN_RATE_DPS, p.default_turn_rate);
  clamped |= take_u(cmd, "watchdog_ms",     50u, HARDCAP_WATCHDOG_MS,       p.watchdog_ms);
  clamped |= take_u(cmd, "drive_expiry_ms", 50u, HARDCAP_DRIVE_EXPIRY_MS,   p.drive_expiry_ms);
  clamped |= take_u(cmd, "manual_idle_return_secs", 0u, 60u,               p.manual_idle_return_secs);
  if (cmd["manual_autoreturn"].is<bool>()) p.manual_autoreturn = cmd["manual_autoreturn"].as<bool>();

  // Drive tuning (real HW): per-wheel PID gains + calibration geometry. Clamped to
  // safe, physically-plausible ranges so a bad push can't divide-by-zero, invert the
  // loop (negative gains), or scale odometry into nonsense. Absent keys keep their
  // current value. NB: a geometry change takes effect immediately and re-scales an
  // in-flight finite command's progress — change geometry at IDLE (the bench tool
  // refuses calibration edits unless the base is idle).
  clamped |= take_f(cmd, "kp", 0.0f, 100000.0f, p.kp);
  clamped |= take_f(cmd, "ki", 0.0f, 100000.0f, p.ki);
  clamped |= take_f(cmd, "kd", 0.0f, 100000.0f, p.kd);
  clamped |= take_f(cmd, "kff",      0.0f, 100000.0f, p.kff);
  clamped |= take_f(cmd, "min_duty", 0.0f, (float)PWM_DUTY_MAX, p.min_duty);
  clamped |= take_f(cmd, "breakaway_duty", 0.0f, (float)PWM_DUTY_MAX, p.breakaway_duty);
  clamped |= take_f(cmd, "accel_lin", 0.05f, 20.0f, p.accel_lin);   // m/s^2  (0 would freeze teleop)
  clamped |= take_f(cmd, "accel_ang", 0.05f, 50.0f, p.accel_ang);   // rad/s^2
  clamped |= take_f(cmd, "counts_per_meter", 1000.0f, 1.0e6f, p.counts_per_meter);
  clamped |= take_f(cmd, "track_width_m",    0.05f,   2.0f,   p.track_width_m);

  // Hallway steering assist (manual forward drive).
  if (cmd["assist_enabled"].is<bool>()) p.assist_enabled = cmd["assist_enabled"].as<bool>();
  clamped |= take_f(cmd, "assist_engage_mm", 0.0f, 2000.0f, p.assist_engage_mm);
  clamped |= take_f(cmd, "assist_gain",      0.0f, 20.0f,   p.assist_gain);

  // Keep zones sane: stop_zone must be < slow_zone.
  if (p.stop_zone_m >= p.slow_zone_m) { p.stop_zone_m = p.slow_zone_m * 0.5f; clamped = true; }

  LOCK_STATE();
  g_ctx.params = p;
  UNLOCK_STATE();
  out = p;
  return clamped;
}
