// Measured COME arrival and comfortable approach speed. Hard safety is separate.
#pragma once
#include <math.h>
#include <stdint.h>

struct ComeArrival {
  static float range_m(int16_t left_mm, int16_t right_mm) {
    int16_t near = 32767;
    if (left_mm > 0 && left_mm < near) near = left_mm;
    if (right_mm > 0 && right_mm < near) near = right_mm;
    return near == 32767 ? -1.f : near * .001f;
  }
  static bool arrived(float range, float stop_at) {
    return range > 0.f && range <= stop_at + .03f;
  }
  static float speed(float range, float stop_at, float cruise, float accel) {
    if (range <= 0.f || arrived(range, stop_at)) return 0.f;
    // Begin deceleration far enough out to shed cruise speed at normal slew.
    float remaining = fmaxf(0.f, range - stop_at - .03f);
    float limit = sqrtf(2.f * fmaxf(.05f, accel) * remaining);
    return fminf(cruise, fmaxf(.04f, limit));
  }
};
