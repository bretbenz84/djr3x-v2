// Heading restoration for COME only. Obstacle avoidance always has priority.
#pragma once
#include <math.h>

struct TravelHeading {
  bool initialized = false;
  bool valid = false;
  float target_rad = 0;
  float clear_secs = 0;

  float correction(float yaw_rad, bool imu_ok, float dt,
                   bool corridor_clear, float avoidance) {
    if (!initialized) {
      initialized = true;
      valid = imu_ok && isfinite(yaw_rad);
      target_rad = yaw_rad;
    }
    // Never change reference frames, or adopt the deflected bearing on recovery.
    if (!imu_ok || !isfinite(yaw_rad)) valid = false;
    if (!corridor_clear || fabsf(avoidance) > 0.01f) {
      clear_secs = 0;
      return avoidance;
    }
    clear_secs += fmaxf(0, fminf(dt, 0.1f));
    if (!valid || clear_secs < 0.4f) return avoidance;
    float error = remainderf(target_rad - yaw_rad, 6.28318530718f);
    if (fabsf(error) < 0.035f) return 0;  // two-degree deadband
    // Gentle return while rolling; ordinary accel and reflex gates still follow.
    return fmaxf(-0.25f, fminf(0.25f, 1.2f * error));
  }
};
