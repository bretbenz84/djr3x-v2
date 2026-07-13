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

struct LaunchState {
  bool  rolling = false;
  float moving_s = 0.0f;
  float stalled_s = 0.0f;
};
static LaunchState s_launch_l, s_launch_r;

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
  s_launch_l = LaunchState();
  s_launch_r = LaunchState();
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
  // Standard REP-103 heading delta (right wheel ahead of left = turning CCW/left), in
  // lockstep with the standard mixing in hal_drive_velocity, so +d_theta = physical
  // CCW/left and odometry heading tracks the real world (voice/D-pad/autonomous turns).
  // History: this was (d_l - d_r) while the channels were cross-wired — see the note in
  // hal_drive_velocity; flip BOTH together or closed-loop turns break.
  const float d_theta  = (d_r - d_l) / track;

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
  const float deriv = (dt > 1e-4f) ? (err - eprev) / dt : 0.0f;
  eprev = err;
  const float ff = kff * target + min_duty * (target >= 0.0f ? 1.0f : -1.0f);
  const float base = ff + kp * err + kd * deriv;
  const float next_i = clampf(integ + ki * err * dt,
                              -WHEEL_PID_I_CLAMP, WHEEL_PID_I_CLAMP);
  const float candidate = base + next_i;
  // Conditional integration: do not wind the integrator farther into saturation.
  // It may still integrate in the opposite direction, which unwinds it promptly.
  if (!((candidate > (float)PWM_DUTY_MAX && err > 0.0f) ||
        (candidate < -(float)PWM_DUTY_MAX && err < 0.0f))) {
    integ = next_i;
  }
  const float u = base + integ;
  return (int)clampf(u, -(float)PWM_DUTY_MAX, (float)PWM_DUTY_MAX);
}

// Stateful launch detector. The high breakaway tier stays active through brief
// encoder motion and is released only after sustained rolling. Once rolling, a
// single quantized zero-speed tick does not re-trigger a kick; a genuine restall does.
static bool needs_breakaway(float target, float meas, float dt, LaunchState& s) {
  if (fabsf(target) < WHEEL_STOP_EPS_MS) {
    s = LaunchState();
    return false;
  }
  const bool moving = fabsf(meas) >= WHEEL_STALLED_EPS_MS;
  if (!s.rolling) {
    s.moving_s = moving ? (s.moving_s + dt) : 0.0f;
    if (s.moving_s >= WHEEL_LAUNCH_CONFIRM_S) {
      s.rolling = true;
      s.stalled_s = 0.0f;
    }
  } else {
    s.stalled_s = moving ? 0.0f : (s.stalled_s + dt);
    if (s.stalled_s >= WHEEL_RESTALL_CONFIRM_S) {
      s.rolling = false;
      s.moving_s = 0.0f;
    }
  }
  return !s.rolling;
}

void hal_drive_velocity(float lin, float ang, float dt, bool pivot_steer, float pivot_blend) {
  // Caller holds the state lock — read the runtime params as a consistent snapshot.
  const float track = g_ctx.params.track_width_m;
  const float kp = g_ctx.params.kp, ki = g_ctx.params.ki, kd = g_ctx.params.kd;
  const float kff = g_ctx.params.kff, min_duty = g_ctx.params.min_duty;
  // Differential-drive kinematics (REP-103: +lin forward, +ang CCW/left): a CCW/left
  // turn slows/reverses the LEFT wheel and speeds the RIGHT. STANDARD mixing — kept in
  // lockstep with d_theta in hal_read_odom. History (don't re-negate): an earlier build
  // of the base had the drive channels cross-wired to opposite physical sides and
  // compensated by negating the angular term here + in odometry; the base has since been
  // wired straight per pins.h (proven by the per-wheel `wheel` bench test 2026-07-11:
  // PIN_L_* spins the physical left wheel), which made that negation reverse all turning
  // (stick fwd+left drove fwd+right). Per-wheel direction is MOTOR_SIGN_*'s job, not this.
  float v_l = lin - ang * (track * 0.5f);
  float v_r = lin + ang * (track * 0.5f);

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

  // Preserve the requested curvature when combined linear+angular input asks an
  // individual wheel to exceed the drivetrain's physical speed. Without this,
  // only the outside wheel saturates and full-stick steering becomes unexpectedly
  // shallow while its PID integrator winds up.
  const float peak = fmaxf(fabsf(v_l), fabsf(v_r));
  if (peak > WHEEL_TARGET_MAX_MS) {
    const float scale = WHEEL_TARGET_MAX_MS / peak;
    v_l *= scale;
    v_r *= scale;
  }

  // Energize only when something should move (commanded OR still rolling, so we
  // actively brake a coasting wheel to a stop before disabling it).
  const bool want_move =
      fabsf(v_l) >= WHEEL_STOP_EPS_MS || fabsf(v_r) >= WHEEL_STOP_EPS_MS ||
      fabsf(s_vmeas_l) >= WHEEL_STOP_EPS_MS || fabsf(s_vmeas_r) >= WHEEL_STOP_EPS_MS;
  if (!want_move) { hal_motors_off(); return; }

  motors_enable(true);
  // Pivot regime: opposite-sign wheel targets mean both tires scrub sideways — a far
  // higher breakaway threshold than rolling (measured: spins stall at duties that move
  // the base fine in a straight line). Each wheel gets the big spin kick ONLY while it
  // is measurably stalled — one tick after it rolls, the kick drops back to the gentle
  // rolling value (a constant large kick would overspeed the spin ~3x after breakaway;
  // stall-gating makes it breakaway torque pulses, and the integrator carries whatever
  // sustained load the surface actually needs). Arcade-clamped turns floor the inside
  // wheel at 0, so they stay in the rolling regime and keep the gentle kick.
  // Duty-floor tiers per wheel (stall-gated; surface-mode + params, context.h):
  //   pivot + STALLED    -> spin breakaway (break static scrub loose)
  //   pivot + ROLLING    -> spin run floor (carry sustained sideways-scrub drag)
  //   straight + STALLED -> straight breakaway (params.breakaway_duty — the full-weight
  //                         robot needs a substantial punch to leave a dead stop; a low
  //                         command otherwise just hums below static friction)
  //   straight + ROLLING -> min_duty (gentle floor — keeps low-speed regulation honest:
  //                         a big CONSTANT floor would overshoot creep targets)
  const bool pivot = (v_l * v_r < -1e-6f);
  const float spin_kick = g_ctx.spin_breakaway_duty;
  const float spin_run  = g_ctx.spin_run_duty;
  const float straight_kick = fmaxf(min_duty, g_ctx.params.breakaway_duty);
  const bool launch_l = needs_breakaway(v_l, s_vmeas_l, dt, s_launch_l);
  const bool launch_r = needs_breakaway(v_r, s_vmeas_r, dt, s_launch_r);
  const float kick_l = launch_l
      ? (pivot ? fmaxf(min_duty, spin_kick) : straight_kick)
      : (pivot ? fmaxf(min_duty, spin_run)  : min_duty);
  const float kick_r = launch_r
      ? (pivot ? fmaxf(min_duty, spin_kick) : straight_kick)
      : (pivot ? fmaxf(min_duty, spin_run)  : min_duty);
  // PID runs in the forward=+ convention; MOTOR_SIGN_* maps its effort onto each
  // H-bridge, so a wheel that spins backwards is fixed in software, not by rewiring.
  const int duty_l = MOTOR_SIGN_L * wheel_pid(v_l, s_vmeas_l, s_i_l, s_eprev_l, dt, kp, ki, kd, kff, kick_l);
  const int duty_r = MOTOR_SIGN_R * wheel_pid(v_r, s_vmeas_r, s_i_r, s_eprev_r, dt, kp, ki, kd, kff, kick_r);
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

// Drive ONE wheel's H-bridge at a raw signed fraction of full duty, bypassing BOTH the
// differential kinematics AND the velocity PID — the bring-up diagnostic for "is this
// wheel wired, and does it spin the right way?". Deliberately open-loop so a mis-wired
// or unread encoder cannot corrupt the test (unlike hal_drive_velocity, whose PID would
// fight a bad encoder). side: 0 = left, 1 = right. frac: signed, -1..1; + = that wheel
// FORWARD per its MOTOR_SIGN_* (so a wrong physical direction means the motor leads /
// MOTOR_SIGN are off). The magnitude is floored at the stiction breakaway (params.min_duty)
// so a small frac still turns a free wheel on a stand, and clamped to PWM_DUTY_MAX; the
// OTHER wheel is held off. Caller (control_tick) holds the state lock, so writing
// g_ctx.wheels for telemetry is safe (mirrors hal_drive_velocity).
void hal_drive_wheel_raw(int side, float frac) {
  frac = clampf(frac, -1.0f, 1.0f);
  int duty = 0;
  if (fabsf(frac) > 1e-3f) {
    float mag = fabsf(frac) * (float)PWM_DUTY_MAX;
    if (mag < g_ctx.params.min_duty) mag = g_ctx.params.min_duty;   // stiction breakaway floor
    duty = (int)clampf(mag, 0.0f, (float)PWM_DUTY_MAX);
    if (frac < 0.0f) duty = -duty;
  }
  motors_enable(true);
  if (side == 0) {                                   // LEFT wheel; right held off
    const int d = MOTOR_SIGN_L * duty;
    apply_wheel_duty(PIN_L_RPWM, PIN_L_LPWM, d);
    apply_wheel_duty(PIN_R_RPWM, PIN_R_LPWM, 0);
    g_ctx.wheels.dl = (int16_t)d; g_ctx.wheels.dr = 0;
  } else {                                           // RIGHT wheel; left held off
    const int d = MOTOR_SIGN_R * duty;
    apply_wheel_duty(PIN_R_RPWM, PIN_R_LPWM, d);
    apply_wheel_duty(PIN_L_RPWM, PIN_L_LPWM, 0);
    g_ctx.wheels.dl = 0; g_ctx.wheels.dr = (int16_t)d;
  }
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
