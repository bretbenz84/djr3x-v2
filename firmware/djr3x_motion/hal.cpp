#include "hal.h"

#if MOTION_HW_PRESENT
// ===========================================================================
// REAL HARDWARE — Phase 1 closed-loop drive base.
//   - 2× BTS7960 H-bridge via LEDC PWM (RPWM/LPWM per wheel) + enable lines.
//   - 2× Hall quadrature encoder via the PCNT peripheral (ESP32Encoder, x4).
//   - Differential-drive kinematics + per-wheel velocity PID on encoder speed.
//   - Odometry integrated from encoder deltas.
// Pins in pins.h, measured/tuned constants in calib.h.
// ToF (8 radial sensors: 4× VL53L0X + 4× VL53L1X) lives in tof.cpp — see
// hal_read_tof() — gated by MOTION_TOF_PRESENT; obstacle avoidance is inactive until
// that subsystem is wired and the build enables it (docs/motion_system.md §6).
// ===========================================================================
#include "pins.h"
#include "calib.h"
#include <Arduino.h>
#include <math.h>
#include <ESP32Encoder.h>

// LEDC PWM helpers — one pin-based interface across Arduino-ESP32 core versions.
// Core 3.x replaced the channel-based LEDC API (ledcSetup / ledcAttachPin / ledcWrite
// by channel) with a pin-based one (ledcAttach / ledcWrite by pin). We support BOTH so
// the firmware builds whether a machine has the legacy 2.x core or the current 3.x core
// — the rest of this file only ever calls pwm_attach(pin) / pwm_write(pin, duty).
#if ESP_ARDUINO_VERSION >= ESP_ARDUINO_VERSION_VAL(3, 0, 0)
static inline void pwm_attach(int pin) { ledcAttach(pin, PWM_FREQ_HZ, PWM_RES_BITS); }
static inline void pwm_write(int pin, int duty) { ledcWrite(pin, duty); }
#else
// 2.x: each PWM pin maps to a fixed LEDC channel; attach + write go through the channel.
static int pwm_chan(int pin) {
  if (pin == PIN_L_RPWM) return 0;
  if (pin == PIN_L_LPWM) return 1;
  if (pin == PIN_R_RPWM) return 2;
  return 3;                                  // PIN_R_LPWM
}
static inline void pwm_attach(int pin) {
  const int ch = pwm_chan(pin);
  ledcSetup(ch, PWM_FREQ_HZ, PWM_RES_BITS);
  ledcAttachPin(pin, ch);
}
static inline void pwm_write(int pin, int duty) { ledcWrite(pwm_chan(pin), duty); }
#endif

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
  if (duty >= 0) { pwm_write(rpwm_pin, duty);  pwm_write(lpwm_pin, 0); }
  else           { pwm_write(rpwm_pin, 0);     pwm_write(lpwm_pin, -duty); }
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

  // LEDC PWM for the four half-bridge inputs; attach each at the configured
  // freq/resolution, all starting at 0 duty.
  pwm_attach(PIN_L_RPWM);
  pwm_attach(PIN_L_LPWM);
  pwm_attach(PIN_R_RPWM);
  pwm_attach(PIN_R_LPWM);
  pwm_write(PIN_L_RPWM, 0); pwm_write(PIN_L_LPWM, 0);
  pwm_write(PIN_R_RPWM, 0); pwm_write(PIN_R_LPWM, 0);

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
  g_ctx.wheels.vl = s_vmeas_l;   // telemetry diag (caller holds the state lock)
  g_ctx.wheels.vr = s_vmeas_r;

  const float d_center = 0.5f * (d_l + d_r);
  // Negated (d_l - d_r, not d_r - d_l) to match the swapped-sides motor mixing in
  // hal_drive_velocity, so +d_theta = physical CCW/left (REP-103) and odometry heading
  // tracks the real world (keeps voice/D-pad/autonomous turns correct).
  const float d_theta  = (d_l - d_r) / track;

  out.theta += d_theta;
  while (out.theta >  (float)M_PI)  out.theta -= 2.0f * (float)M_PI;
  while (out.theta <= -(float)M_PI) out.theta += 2.0f * (float)M_PI;
  out.x += d_center * cosf(out.theta);
  out.y += d_center * sinf(out.theta);
  out.lin = d_center * inv_dt;
  out.ang = d_theta  * inv_dt;
}

// One wheel's velocity controller: target m/s -> signed duty. Feedforward + stiction
// kick + PID trim. Gains are runtime-tunable (g_ctx.params); i_clamp stays a
// compile-time safety bound. Updates integ/eprev.
//
// u = KFF*target                      feedforward: ~right duty the instant a speed is
//                                     commanded (no waiting for the integrator to wind up)
//   + MIN_DUTY*sign(target)          stiction breakaway kick in the travel direction
//   + kp*err + integ + kd*deriv      closed-loop trim on the encoder-measured error
// Without the first two terms the loop starts every move from zero duty, so low
// speeds sit below breakaway friction (weak + slow to start) and duty only scales up
// as the integrator climbs (strong only once fast) — the reported feel. With them the
// wheel is strong and responsive from the first tick; the PID just corrects.
static int wheel_pid(float target, float meas, float& integ, float& eprev, float dt,
                     float kp, float ki, float kd, float kff, float min_duty) {
  if (fabsf(target) < WHEEL_STOP_EPS_MS) {
    integ = 0; eprev = 0;            // commanded stop: don't chase, drop windup
    return 0;
  }
  const float err = target - meas;
  integ += ki * err * dt;
  integ = clampf(integ, -WHEEL_PID_I_CLAMP, WHEEL_PID_I_CLAMP);   // anti-windup
  const float deriv = (dt > 1e-4f) ? (err - eprev) / dt : 0.0f;
  eprev = err;
  const float ff = kff * target + min_duty * (target >= 0.0f ? 1.0f : -1.0f);
  const float u  = ff + kp * err + integ + kd * deriv;
  return (int)clampf(u, -(float)PWM_DUTY_MAX, (float)PWM_DUTY_MAX);
}

void hal_drive_velocity(float lin, float ang, float dt, bool pivot_steer, float pivot_blend) {
  // Caller holds the state lock — read the runtime params as a consistent snapshot.
  const float track = g_ctx.params.track_width_m;
  const float kp = g_ctx.params.kp, ki = g_ctx.params.ki, kd = g_ctx.params.kd;
  const float kff = g_ctx.params.kff, min_duty = g_ctx.params.min_duty;
  // Differential-drive kinematics (REP-103: +lin forward, +ang CCW/left). The drive
  // channels are wired to the OPPOSITE physical sides (verified on the bench: a +ang
  // command spun the base CW/right while forward/back were correct), so the angular term
  // is negated here — kept in lockstep with the same negation in hal_read_odom — to make
  // +ang = physical CCW/left WITHOUT swapping the pin map (pins.h stays as-wired).
  float v_l = lin + ang * (track * 0.5f);
  float v_r = lin - ang * (track * 0.5f);

  // Joystick steering (teleop only) — a smooth MORPH between two mixings, driven by
  // pivot_blend (0..1, from the fwd-stick fraction in gamepad.cpp):
  //   • blend 0 (pure left/right, no forward/back): SPIN IN PLACE. Raw differential
  //     mix — the inside wheel runs backward and the base rotates on the spot.
  //   • blend 1 (clearly translating): ARCADE STEER. Each wheel is floored at zero
  //     against the travel direction, so a turn only slows the inside wheel — it
  //     never reverses from steering, only from the stick itself being pulled back.
  //   • between: per-wheel lerp of the two, so tilting the stick forward out of a
  //     spin tightens smoothly into an arc (no regime snap at a threshold).
  // NOT applied to autonomous paths (control_tick passes pivot_steer=false) — finite
  // turns spin via CMD_TURN and the Mac's velocity drive keeps plain mixing.
  if (pivot_steer && pivot_blend > 0.0f) {
    float a_l = v_l, a_r = v_r;              // the arcade (clamped) mix
    if (lin >= 0.0f) { if (a_l < 0.0f) a_l = 0.0f; if (a_r < 0.0f) a_r = 0.0f; }
    else             { if (a_l > 0.0f) a_l = 0.0f; if (a_r > 0.0f) a_r = 0.0f; }
    v_l += (a_l - v_l) * pivot_blend;        // lerp raw spin mix -> arcade mix
    v_r += (a_r - v_r) * pivot_blend;
  }

  // Energize only when something should move (commanded OR still rolling, so we
  // actively brake a coasting wheel to a stop before disabling it).
  const bool want_move =
      fabsf(v_l) >= WHEEL_STOP_EPS_MS || fabsf(v_r) >= WHEEL_STOP_EPS_MS ||
      fabsf(s_vmeas_l) >= WHEEL_STOP_EPS_MS || fabsf(s_vmeas_r) >= WHEEL_STOP_EPS_MS;
  if (!want_move) { hal_motors_off(); return; }

  motors_enable(true);
  // PID runs in the forward=+ convention; MOTOR_SIGN_* maps its effort onto each
  // H-bridge, so a wheel that spins backwards is fixed in software, not by rewiring.
  const int duty_l = MOTOR_SIGN_L * wheel_pid(v_l, s_vmeas_l, s_i_l, s_eprev_l, dt, kp, ki, kd, kff, min_duty);
  const int duty_r = MOTOR_SIGN_R * wheel_pid(v_r, s_vmeas_r, s_i_r, s_eprev_r, dt, kp, ki, kd, kff, min_duty);
  apply_wheel_duty(PIN_L_RPWM, PIN_L_LPWM, duty_l);
  apply_wheel_duty(PIN_R_RPWM, PIN_R_LPWM, duty_r);
  g_ctx.wheels.dl = (int16_t)duty_l;   // telemetry diag (caller holds the state lock)
  g_ctx.wheels.dr = (int16_t)duty_r;
}

void hal_motors_off() {
  pwm_write(PIN_L_RPWM, 0); pwm_write(PIN_L_LPWM, 0);
  pwm_write(PIN_R_RPWM, 0); pwm_write(PIN_R_LPWM, 0);
  motors_enable(false);
  reset_pid();
  g_ctx.wheels.dl = 0; g_ctx.wheels.dr = 0;   // duties off (caller holds the state lock)
}

// hal_read_tof()/hal_tof_init() live in tof.cpp — they are gated by MOTION_TOF_PRESENT
// independently of the motor drivers (the base can drive before the ToF is wired).

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
#endif
