#include "hal.h"

#if MOTION_HW_PRESENT
// ===========================================================================
// REAL HARDWARE — Phase 1 closed-loop drive base.
//   - 2× BTS7960 H-bridge via LEDC PWM (RPWM/LPWM per wheel) + enable lines.
//   - 2× Hall quadrature encoder via the PCNT peripheral (ESP32Encoder, x4).
//   - Differential-drive kinematics + per-wheel velocity PID on encoder speed.
//   - Odometry integrated from encoder deltas.
// Pins in pins.h, measured/tuned constants in calib.h.
// ToF (5× VL53L0X) is NOT here yet — see hal_read_tof() — so obstacle avoidance
// is inactive until that subsystem is wired and its addressing scheme is chosen
// (docs/motion_system.md §6).
// ===========================================================================
#include "pins.h"
#include "calib.h"
#include <Arduino.h>
#include <math.h>
#include <ESP32Encoder.h>

// LEDC PWM is driven through the core-3.x pin-based API: ledcAttach() allocates a
// channel per pin under the hood, and duty is written by GPIO (ledcWrite(pin,duty)).
// (Arduino-ESP32 3.x removed the old channel-based ledcSetup/ledcAttachPin.)

static ESP32Encoder encL, encR;

// Per-tick measurement state.
static int64_t s_prev_l = 0, s_prev_r = 0;   // encoder counts at the previous tick
static float   s_vmeas_l = 0, s_vmeas_r = 0; // measured wheel speeds (m/s) this tick
static bool    s_motors_enabled = false;

// Per-wheel PID state (duty units).
static float s_i_l = 0, s_i_r = 0;           // integral accumulators
static float s_eprev_l = 0, s_eprev_r = 0;   // previous error (for D term)

static inline void motors_enable(bool en) {
  if (en == s_motors_enabled) return;        // only toggle the GPIO on a real change
  digitalWrite(PIN_L_EN, en ? HIGH : LOW);
  digitalWrite(PIN_R_EN, en ? HIGH : LOW);
  s_motors_enabled = en;
}

// Apply a signed duty (-MAX..+MAX) to one wheel by PWM-ing exactly one half of
// its H-bridge (the other at 0). Forward = RPWM, reverse = LPWM; never both.
static inline void apply_wheel_duty(int rpwm_pin, int lpwm_pin, int duty) {
  if (duty >= 0) { ledcWrite(rpwm_pin, duty);  ledcWrite(lpwm_pin, 0); }
  else           { ledcWrite(rpwm_pin, 0);     ledcWrite(lpwm_pin, -duty); }
}

static void reset_pid() {
  s_i_l = s_i_r = 0;
  s_eprev_l = s_eprev_r = 0;
}

void hal_init() {
  // Motor enables: start DISABLED so the base is boot-safe (motors coast, no
  // drive until an explicit command energizes them).
  pinMode(PIN_L_EN, OUTPUT);
  pinMode(PIN_R_EN, OUTPUT);
  digitalWrite(PIN_L_EN, LOW);
  digitalWrite(PIN_R_EN, LOW);
  s_motors_enabled = false;

  // LEDC PWM for the four half-bridge inputs; all at 0 duty. ledcAttach (core 3.x)
  // allocates a channel per pin at the given freq/resolution.
  ledcAttach(PIN_L_RPWM, PWM_FREQ_HZ, PWM_RES_BITS);
  ledcAttach(PIN_L_LPWM, PWM_FREQ_HZ, PWM_RES_BITS);
  ledcAttach(PIN_R_RPWM, PWM_FREQ_HZ, PWM_RES_BITS);
  ledcAttach(PIN_R_LPWM, PWM_FREQ_HZ, PWM_RES_BITS);
  ledcWrite(PIN_L_RPWM, 0); ledcWrite(PIN_L_LPWM, 0);
  ledcWrite(PIN_R_RPWM, 0); ledcWrite(PIN_R_LPWM, 0);

  // Encoders: full quadrature (x4). Internal weak pull-ups so a disconnected or
  // floating input doesn't generate phantom counts during incremental bring-up.
  ESP32Encoder::useInternalWeakPullResistors = puType::up;
  encL.attachFullQuad(PIN_ENC_L_A, PIN_ENC_L_B);
  encR.attachFullQuad(PIN_ENC_R_A, PIN_ENC_R_B);
  encL.clearCount();
  encR.clearCount();
  s_prev_l = 0; s_prev_r = 0;
  s_vmeas_l = s_vmeas_r = 0;
  reset_pid();
}

void hal_read_odom(Odom& out, float dt) {
  // Caller (control_tick) holds the state lock, so reading the runtime-tunable
  // params here is a consistent snapshot. apply_config clamps both > 0, so the
  // divides below are safe.
  const float cpm   = g_ctx.params.counts_per_meter;
  const float track = g_ctx.params.track_width_m;
  const int64_t cl = encL.getCount();
  const int64_t cr = encR.getCount();
  // Signed wheel travel (metres) since the previous tick.
  const float d_l = ENC_SIGN_L * (float)(cl - s_prev_l) / cpm;
  const float d_r = ENC_SIGN_R * (float)(cr - s_prev_r) / cpm;
  s_prev_l = cl;
  s_prev_r = cr;

  // Floor well below the 100 Hz control period (0.01 s) but above any sane tick,
  // so a pathological dt can't divide-by-zero or amplify encoder noise into the
  // velocity estimate. Position (x/y/theta) integrates from the deltas directly
  // and is unaffected.
  const float inv_dt = (dt > 1e-4f) ? (1.0f / dt) : 0.0f;
  s_vmeas_l = d_l * inv_dt;
  s_vmeas_r = d_r * inv_dt;

  const float d_center = 0.5f * (d_l + d_r);
  const float d_theta  = (d_r - d_l) / track;           // +d_theta = CCW (REP-103)

  out.theta += d_theta;
  while (out.theta >  (float)M_PI)  out.theta -= 2.0f * (float)M_PI;
  while (out.theta <= -(float)M_PI) out.theta += 2.0f * (float)M_PI;
  out.x += d_center * cosf(out.theta);
  out.y += d_center * sinf(out.theta);
  out.lin = d_center * inv_dt;
  out.ang = d_theta  * inv_dt;
}

// One wheel's velocity PID: target m/s -> signed duty. Gains are runtime-tunable
// (g_ctx.params); i_clamp stays a compile-time safety bound. Updates integ/eprev.
static int wheel_pid(float target, float meas, float& integ, float& eprev, float dt,
                     float kp, float ki, float kd) {
  if (fabsf(target) < WHEEL_STOP_EPS_MS) {
    integ = 0; eprev = 0;            // commanded stop: don't chase, drop windup
    return 0;
  }
  const float err = target - meas;
  integ += ki * err * dt;
  integ = clampf(integ, -WHEEL_PID_I_CLAMP, WHEEL_PID_I_CLAMP);   // anti-windup
  const float deriv = (dt > 1e-4f) ? (err - eprev) / dt : 0.0f;
  eprev = err;
  const float u = kp * err + integ + kd * deriv;
  return (int)clampf(u, -(float)PWM_DUTY_MAX, (float)PWM_DUTY_MAX);
}

void hal_drive_velocity(float lin, float ang, float dt) {
  // Caller holds the state lock — read the runtime params as a consistent snapshot.
  const float track = g_ctx.params.track_width_m;
  const float kp = g_ctx.params.kp, ki = g_ctx.params.ki, kd = g_ctx.params.kd;
  // Differential-drive kinematics (REP-103: +lin forward, +ang CCW/left).
  const float v_l = lin - ang * (track * 0.5f);
  const float v_r = lin + ang * (track * 0.5f);

  // Energize only when something should move (commanded OR still rolling, so we
  // actively brake a coasting wheel to a stop before disabling it).
  const bool want_move =
      fabsf(v_l) >= WHEEL_STOP_EPS_MS || fabsf(v_r) >= WHEEL_STOP_EPS_MS ||
      fabsf(s_vmeas_l) >= WHEEL_STOP_EPS_MS || fabsf(s_vmeas_r) >= WHEEL_STOP_EPS_MS;
  if (!want_move) { hal_motors_off(); return; }

  motors_enable(true);
  const int duty_l = wheel_pid(v_l, s_vmeas_l, s_i_l, s_eprev_l, dt, kp, ki, kd);
  const int duty_r = wheel_pid(v_r, s_vmeas_r, s_i_r, s_eprev_r, dt, kp, ki, kd);
  apply_wheel_duty(PIN_L_RPWM, PIN_L_LPWM, duty_l);
  apply_wheel_duty(PIN_R_RPWM, PIN_R_LPWM, duty_r);
}

void hal_motors_off() {
  ledcWrite(PIN_L_RPWM, 0); ledcWrite(PIN_L_LPWM, 0);
  ledcWrite(PIN_R_RPWM, 0); ledcWrite(PIN_R_LPWM, 0);
  motors_enable(false);
  reset_pid();
}

void hal_read_tof(TofMm& out) {
  // Phase-1 ToF (5× VL53L0X) is not wired/implemented yet, so report a clear
  // room and let the reflex/zone logic run. OBSTACLE AVOIDANCE IS INACTIVE until
  // the real driver lands (XSHUT sequencing or TCA9548A mux — docs §6.1).
  // down=60 mm => floor present, well under the cliff threshold (no false cliff).
  out.fl = out.fc = out.fr = out.rear = 1500;
  out.down = 60;
}

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
