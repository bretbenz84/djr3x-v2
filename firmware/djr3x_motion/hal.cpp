#include "hal.h"

#if MOTION_HW_PRESENT
// ===========================================================================
// REAL HARDWARE — fill these in as peripherals are wired. Suggested layout in
// docs/motion_system.md §5 (BTS7960 ×2) and §6 (VL53L0X ×5).
// ===========================================================================
//   - LEDC PWM on RPWM/LPWM per motor; convert (lin,ang) -> per-wheel duty via
//     the differential-drive kinematics + per-wheel PID on encoder speed.
//   - Hall quadrature encoders on interrupt-capable pins -> COUNTS_PER_METER.
//   - 5× VL53L0X: XSHUT-sequenced unique I2C addresses (or TCA9548A mux).
// Until then this block is intentionally empty so the build still links.
void hal_init() { /* TODO: init LEDC, encoder ISRs, I2C + VL53L0X addressing */ }
void hal_apply_velocity(float lin, float ang) { (void)lin; (void)ang; /* TODO: PWM */ }
void hal_read_tof(TofMm& out) { /* TODO: read 5 sensors into out */ }

#else
// ===========================================================================
// STUB — no peripherals. Motors are a no-op; ToF reports a clear room so the
// reflex/zone logic stays in CLEAR and nothing blocks. The plant model in
// control.cpp synthesizes odometry from the commanded velocity.
// ===========================================================================
void hal_init() {
  // Nothing to initialize in the stub.
}

void hal_apply_velocity(float lin, float ang) {
  (void)lin;
  (void)ang;   // no motors wired; the plant model integrates these in control.cpp
}

void hal_read_tof(TofMm& out) {
  out.fl   = 1500;
  out.fc   = 1500;
  out.fr   = 1500;
  out.rear = 1500;
  out.down = 60;     // floor present (~60 mm), well under the cliff threshold
}
#endif
