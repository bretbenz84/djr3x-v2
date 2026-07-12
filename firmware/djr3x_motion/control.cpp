#include "control.h"
#include "proto_io.h"
#include "hal.h"
#include <math.h>

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif
#define DEG2RAD(d) ((d) * (float)M_PI / 180.0f)

// Stub plant dynamics (replaced by the real motor/encoder loop when HW lands).
#define PLANT_ACCEL_LIN 0.6f    // m/s^2
#define PLANT_ACCEL_ANG 3.0f    // rad/s^2
#define MOTION_EPS      0.005f

static inline float wrap_pi(float a) {
  while (a > (float)M_PI)  a -= 2.0f * (float)M_PI;
  while (a <= -(float)M_PI) a += 2.0f * (float)M_PI;
  return a;
}
static inline float signf(float v) { return v >= 0.0f ? 1.0f : -1.0f; }

#if MOTION_HW_PRESENT
// Teleop setpoint slew state (m/s, rad/s). The commanded velocity ramps toward the
// stick target at params.accel_* — symmetric, so release coasts to a stop instead of
// hard dynamic-braking. Owned solely by control_tick (single control task), reset on
// any halt. Only CMD_DRIVE (gamepad/teleop) is slewed; finite move/turn/come stay crisp.
static float s_ramp_lin = 0.0f, s_ramp_ang = 0.0f;
#endif

void control_init() {
  // Nothing yet; plant state lives in g_ctx (zeroed at construction).
}

// Nearest VALID reading of a sensor pair, capped at `cap` — "no wall inside the
// engage distance" and "sensor error" both read as exactly `cap`, so the steering
// assist's imbalance term is zero unless a real wall is actually close.
static int16_t nearest_capped(int16_t a, int16_t b, int16_t cap) {
  int16_t m = cap;
  if (a >= 0 && a < m) m = a;
  if (b >= 0 && b < m) m = b;
  return m;
}

// Hallway steering assist (docs §6.3): while the GAMEPAD drives FORWARD, steer away
// from nearby walls / center between them. Side short pairs give lateral clearance;
// the front long pair adds an anticipatory term (approaching a wall at an angle
// steers toward the open side, so a hallway curve is followed instead of face-planted).
// The correction ADDS to the operator's stick (capped at ASSIST_MAX_ANG_FRAC of
// max_ang, so the human always wins a fight) and rides the same slew/blend path as
// any other teleop command. The Z_STOP reflex still hard-blocks regardless. Returns
// the rad/s correction (+ = steer left, REP-103). Caller holds the state lock.
static float hall_assist_correction(const MotionContext& c, float lin_t) {
  if (!c.params.assist_enabled) return 0.0f;
  if (c.owner != OWNER_MANUAL || c.cmd_mode != CMD_DRIVE) return 0.0f;
  if (lin_t <= ASSIST_MIN_LIN_MS) return 0.0f;      // forward drive only
  if (c.full_override) return 0.0f;                 // operator explicitly bypassing ToF
  const int16_t eng = (int16_t)c.params.assist_engage_mm;
  if (eng <= 0) return 0.0f;
  const int16_t l_side = nearest_capped(c.tof.lf, c.tof.lb, eng);
  const int16_t r_side = nearest_capped(c.tof.rf, c.tof.rb, eng);
  const int16_t l_frnt = nearest_capped(c.tof.fl, -1, eng);
  const int16_t r_frnt = nearest_capped(c.tof.fr, -1, eng);
  // Imbalance in metres: positive (l - r) = the left side is more open (right wall
  // closer) -> steer LEFT (+ang, REP-103) toward the open side; negative mirrors.
  const float imbal_m = ((float)(l_side - r_side)
                         + ASSIST_FRONT_WEIGHT * (float)(l_frnt - r_frnt)) * 0.001f;

  // Close-wall REPULSION (ASSIST_REPEL_MM, ~5 in): a side wall this close pushes back
  // hard on its own, independent of the other side. The imbalance term alone reads
  // ~zero when both walls are equally close (a centered-but-too-narrow gap) and is
  // weak at grazing contact — the field-logged wall scrapes. Both sides inside REPEL
  // cancel proportionally -> net push away from the NEARER wall.
  float repel = 0.0f;
  const int16_t l_min = nearest_capped(c.tof.lf, c.tof.lb, 32767);
  const int16_t r_min = nearest_capped(c.tof.rf, c.tof.rb, 32767);
  if ((float)l_min < ASSIST_REPEL_MM)   // left wall close -> steer RIGHT (-ang)
    repel -= ASSIST_REPEL_GAIN * (ASSIST_REPEL_MM - (float)l_min) * 0.001f;
  if ((float)r_min < ASSIST_REPEL_MM)   // right wall close -> steer LEFT (+ang)
    repel += ASSIST_REPEL_GAIN * (ASSIST_REPEL_MM - (float)r_min) * 0.001f;

  const float corr = c.params.assist_gain * imbal_m + repel;
  const float cap  = ASSIST_MAX_ANG_FRAC * c.params.max_ang;
  return clampf(corr, -cap, cap);
}

// Travel direction a finite command pushes toward (for reflex-block matching).
// turn has no linear travel; swing-side ToF gating for spins is Phase 1.
static MotionDir finite_travel_dir(const FiniteCmd& f) {
  switch (f.kind) {
    case CMD_MOVE: return f.target_dist >= 0 ? DIR_FRONT : DIR_REAR;
    case CMD_COME: return f.come_turning ? DIR_NONE : DIR_FRONT;  // advance is forward
    default:       return DIR_NONE;
  }
}

// ---- control tick: runs entirely under the state lock (race-free), then
// emits any `done` AFTER releasing the lock. -------------------------------
void control_tick(float dt) {
  uint32_t now = millis();

  bool       emitDone = false;
  DoneResult dres     = DONE_COMPLETED;
  uint32_t   dseq     = 0;
  Odom       dodom;

  LOCK_STATE();
  MotionContext& c = g_ctx;

  const bool halted = (c.state == ST_ESTOP || c.state == ST_FAULT || c.state == ST_COMMS_LOST);

  // A halt with a finite command in flight terminates it once.
  if (halted && c.finite.kind != CMD_NONE) {
    emitDone = true; dseq = c.finite.seq; dodom = c.odom;
    dres = (c.state == ST_ESTOP) ? DONE_ESTOPPED : DONE_ABORTED;
    c.finite = FiniteCmd();
    c.cmd_mode = CMD_NONE;
  }

  // A reflex block in the finite command's OWN travel direction terminates it
  // once with done:blocked (contract §7.4). Motion away from the block keeps
  // running (its travel dir won't match blocked_dir). The completion below is
  // skipped because emitDone is now set.
  if (!halted && !emitDone && c.state == ST_BLOCKED && c.finite.kind != CMD_NONE) {
    MotionDir ft = finite_travel_dir(c.finite);
    if (ft != DIR_NONE && ft == c.blocked_dir) {
      emitDone = true; dres = DONE_BLOCKED; dseq = c.finite.seq; dodom = c.odom;
      c.finite = FiniteCmd();
      c.cmd_mode = CMD_NONE;
    }
  }

  // Drive deadman: a stale `drive` setpoint expires -> ramp to stop.
  if (!halted && c.cmd_mode == CMD_DRIVE &&
      (uint32_t)(now - c.drive_set_ms) > c.params.drive_expiry_ms) {
    c.setpoint.lin = 0; c.setpoint.ang = 0; c.cmd_mode = CMD_NONE;
  }

  // Commanded target velocity for this tick.
  float lin_t = 0, ang_t = 0;
  if (!halted) {
    switch (c.cmd_mode) {
      case CMD_DRIVE:
        lin_t = c.setpoint.lin; ang_t = c.setpoint.ang;
        break;
      case CMD_TURN:
        ang_t = signf(c.finite.target_dtheta) * c.finite.rate;
        break;
      case CMD_MOVE:
        lin_t = signf(c.finite.target_dist) * c.finite.speed;
        break;
      case CMD_COME:
        if (c.finite.come_turning) ang_t = signf(c.finite.target_dtheta) * c.finite.rate;
        else                       lin_t = c.finite.speed;   // always forward
        break;
      default: break;
    }
  }

  // Hallway steering assist: nudge the manual forward drive away from nearby walls
  // (centering in a hallway) BEFORE the reflex gate — the assist steers, the reflex
  // still stops. No-op unless MANUAL + CMD_DRIVE + moving forward + a wall inside
  // the engage distance (open rooms and the stub build are exactly zero correction).
  if (!halted) ang_t += hall_assist_correction(c, lin_t);

  // Reflex gating: if blocked in the travel direction, zero that component — UNLESS
  // a gamepad operator is holding full-override (docs §11.4), which deliberately
  // bypasses ToF for nudging through tight spots. estop/fault/comms-loss halts above
  // are NOT bypassed by full_override.
  if (c.state == ST_BLOCKED && !c.full_override) {
    if (c.blocked_dir == DIR_FRONT && lin_t > 0) lin_t = 0;
    if (c.blocked_dir == DIR_REAR  && lin_t < 0) lin_t = 0;
  }

  // Progressive approach slowdown — the SLOW zone actually slows now (field fix
  // 2026-07-11: nothing consumed Z_SLOW, so a straight run at a wall carried full
  // speed to the 0.25 m stop line and momentum did the rest). Inside slow_zone_m
  // toward the obstacle, the commanded speed scales linearly down to ZERO at the
  // stop boundary, so the base sheds speed BEFORE the hard block. Runs on the same
  // fail-open distance read as the reflex (dead pair -> 32767 -> no scaling); the
  // AFTER-assist position keeps the assist's forward-drive gate seeing the raw
  // command. Full-override bypasses, like the reflex. Steering (ang_t) is never
  // scaled — turning away is how you escape a wall.
  if (!halted && !c.full_override && fabsf(lin_t) > 1e-4f) {
    const float stop_m = c.params.stop_zone_m;
    const float slow_m = c.params.slow_zone_m;
    if (slow_m > stop_m + 1e-3f) {
      const int16_t d_near = (lin_t > 0)
          ? nearest_capped(c.tof.fl, c.tof.fr, 32767)
          : nearest_capped(c.tof.rl, c.tof.rr, 32767);
      const float d_m = (float)d_near * 0.001f;
      if (d_m < slow_m)
        lin_t *= clampf((d_m - stop_m) / (slow_m - stop_m), 0.0f, 1.0f);
    }
  }

  // Defensive clamp to caps.
  lin_t = clampf(lin_t, -c.params.max_lin, c.params.max_lin);
  ang_t = clampf(ang_t, -c.params.max_ang, c.params.max_ang);

#if MOTION_HW_PRESENT
  // Real base: odometry comes from the wheel encoders (sense before act). dt is
  // the fixed control period; lin_t/ang_t drive the wheels below.
  hal_read_odom(c.odom, dt);
#else
  // Stub plant: ramp actual velocity toward target (accel-limited)...
  c.odom.lin += clampf(lin_t - c.odom.lin, -PLANT_ACCEL_LIN * dt, PLANT_ACCEL_LIN * dt);
  c.odom.ang += clampf(ang_t - c.odom.ang, -PLANT_ACCEL_ANG * dt, PLANT_ACCEL_ANG * dt);

  // ...and integrate odometry (simple Euler; good enough for the stub).
  c.odom.theta = wrap_pi(c.odom.theta + c.odom.ang * dt);
  c.odom.x += c.odom.lin * cosf(c.odom.theta) * dt;
  c.odom.y += c.odom.lin * sinf(c.odom.theta) * dt;
#endif

  // Finite-command progress.
  if (!halted && !emitDone && c.finite.kind != CMD_NONE) {
    switch (c.finite.kind) {
      case CMD_TURN:
        c.finite.progress_dtheta += fabsf(c.odom.ang) * dt;
        if (c.finite.progress_dtheta >= fabsf(c.finite.target_dtheta)) {
          emitDone = true; dres = DONE_COMPLETED; dseq = c.finite.seq; dodom = c.odom;
          c.finite = FiniteCmd(); c.cmd_mode = CMD_NONE;
        }
        break;
      case CMD_MOVE:
        c.finite.progress_dist += fabsf(c.odom.lin) * dt;
        if (c.finite.progress_dist >= fabsf(c.finite.target_dist)) {
          emitDone = true; dres = DONE_COMPLETED; dseq = c.finite.seq; dodom = c.odom;
          c.finite = FiniteCmd(); c.cmd_mode = CMD_NONE;
        }
        break;
      case CMD_COME:
        if (c.finite.come_turning) {
          c.finite.progress_dtheta += fabsf(c.odom.ang) * dt;
          if (c.finite.progress_dtheta >= fabsf(c.finite.target_dtheta))
            c.finite.come_turning = false;          // heading reached -> advance
        } else {
          c.finite.progress_dist += fabsf(c.odom.lin) * dt;
          float front = c.finite.come_sim_wall - c.finite.progress_dist;  // stub wall
          if (front <= c.finite.come_stop_at) {
            emitDone = true; dres = DONE_COMPLETED; dseq = c.finite.seq; dodom = c.odom;
            c.finite = FiniteCmd(); c.cmd_mode = CMD_NONE;
          }
        }
        break;
      case CMD_WHEEL:
        // Time-bounded (NOT encoder-bounded — the whole point is to work with an
        // unvalidated encoder). Rollover-safe unsigned elapsed compare, like the
        // drive deadman above. On completion the act section below sees CMD_NONE and
        // stops the wheel.
        if ((uint32_t)(now - c.finite.wheel_start_ms) >= c.finite.wheel_ms) {
          emitDone = true; dres = DONE_COMPLETED; dseq = c.finite.seq; dodom = c.odom;
          c.finite = FiniteCmd(); c.cmd_mode = CMD_NONE;
        }
        break;
      default: break;
    }
  }

  // We own only the IDLE<->MOVING transition; safety/dispatch own the rest.
  if (c.state == ST_IDLE || c.state == ST_MOVING) {
    bool moving = (fabsf(c.odom.lin) > MOTION_EPS || fabsf(c.odom.ang) > MOTION_EPS) ||
                  c.cmd_mode != CMD_NONE || c.finite.kind != CMD_NONE;
    c.state = moving ? ST_MOVING : ST_IDLE;
  }

#if MOTION_HW_PRESENT
  // Act: drive the wheels toward the (reflex-gated, clamped) target velocity, or
  // cut the motors entirely on any halt (estop / fault / comms-lost).
  if (halted) {
    s_ramp_lin = 0.0f; s_ramp_ang = 0.0f;   // don't resume mid-ramp after a halt clears
    hal_motors_off();
  } else if (c.cmd_mode == CMD_WHEEL) {
    // Single-wheel bring-up jog: raw duty on ONE wheel, no kinematics/PID. Keep the
    // teleop ramp pinned to zero so a later takeover starts from rest, not mid-jog.
    s_ramp_lin = 0.0f; s_ramp_ang = 0.0f;
    hal_drive_wheel_raw(c.finite.wheel_side, c.finite.wheel_frac);
  } else if (c.cmd_mode == CMD_DRIVE) {
    // Hard halt on a reflex block: CUT the teleop ramp toward the obstacle instead of
    // letting it decay at accel_lin — with the target zeroed the wheels sit enabled at
    // zero duty (BTS7960 both-low = dynamic brake), so the base stops NOW rather than
    // coasting the last stretch into the wall (field fix 2026-07-11).
    if (c.state == ST_BLOCKED && !c.full_override) {
      if (c.blocked_dir == DIR_FRONT && s_ramp_lin > 0) s_ramp_lin = 0;
      if (c.blocked_dir == DIR_REAR  && s_ramp_lin < 0) s_ramp_lin = 0;
    }
    // Teleop: slew the commanded velocity toward the target (accel-limited, symmetric)
    // so a stick push ramps up briskly and a release coasts to a stop rather than
    // stepping to zero and dynamic-braking. Feedforward (in wheel_pid) keeps the ramp
    // responsive; the slew just removes the jerk at both ends.
    const float al = c.params.accel_lin * dt;
    const float aa = c.params.accel_ang * dt;
    s_ramp_lin += clampf(lin_t - s_ramp_lin, -al, al);
    s_ramp_ang += clampf(ang_t - s_ramp_ang, -aa, aa);
    // Joystick steering (owner==MANUAL): pivot_blend (0 spin .. 1 arcade, from the
    // fwd-stick fraction) smoothly morphs the wheel mixing — a pure spin may reverse
    // the inside wheel; as forward is added the reverse allowance eases out until a
    // turn only slows the inside wheel. The Mac's autonomous `drive` (owner==AUTO)
    // keeps plain differential mixing (blend forced to 1, pivot_steer false).
    hal_drive_velocity(s_ramp_lin, s_ramp_ang, dt,
                       c.owner == OWNER_MANUAL, c.setpoint.pivot_blend);
  } else {
    // Autonomous finite move/turn/come (or idle): drive the target directly and keep
    // the ramp synced to it, so a later teleop takeover starts from the real velocity.
    // pivot_steer=false — a finite TURN must spin in place (one wheel reverses).
    s_ramp_lin = lin_t; s_ramp_ang = ang_t;
    hal_drive_velocity(lin_t, ang_t, dt, false, 1.0f);
  }
#else
  // Push velocity to the motor HAL (stub: no-op until wheels are wired).
  hal_apply_velocity(c.odom.lin, c.odom.ang);
#endif

  UNLOCK_STATE();

  if (emitDone) emit_done(dseq, dres, dodom);
}

// ---- Command entry points ------------------------------------------------
// Each supersedes any in-flight finite command (emits done:superseded for it),
// then arms the new motion. Emits happen AFTER releasing the state lock.

static bool begin_finite_locked(uint32_t& sup_seq, Odom& sup_odom) {
  // Caller holds the lock. Returns true if a prior finite cmd must be superseded.
  if (g_ctx.finite.kind != CMD_NONE && g_ctx.finite.kind != CMD_DRIVE) {
    sup_seq = g_ctx.finite.seq; sup_odom = g_ctx.odom; return true;
  }
  return false;
}

void ctl_drive(float lin, float ang, uint32_t seq) {
  bool sup = false; uint32_t sseq = 0; Odom sodom;
  LOCK_STATE();
  sup = begin_finite_locked(sseq, sodom);
  g_ctx.finite = FiniteCmd();
  g_ctx.setpoint.lin = lin; g_ctx.setpoint.ang = ang;
  g_ctx.setpoint.pivot_blend = 1.0f;   // Mac velocity drive: plain differential mixing
  g_ctx.cmd_mode = CMD_DRIVE;
  g_ctx.drive_set_ms = millis();
  g_ctx.cmd_seq = seq;
  if (g_ctx.state == ST_IDLE) g_ctx.state = ST_MOVING;
  UNLOCK_STATE();
  if (sup) emit_done(sseq, DONE_SUPERSEDED, sodom);
}

void ctl_turn(float deg, float rate_dps, uint32_t seq) {
  bool sup = false; uint32_t sseq = 0; Odom sodom;
  LOCK_STATE();
  sup = begin_finite_locked(sseq, sodom);
  FiniteCmd f;
  f.kind = CMD_TURN; f.seq = seq;
  f.target_dtheta = DEG2RAD(deg);
  f.rate = DEG2RAD(rate_dps);
  g_ctx.finite = f;
  g_ctx.cmd_mode = CMD_TURN;
  g_ctx.cmd_seq = seq;
  if (g_ctx.state == ST_IDLE) g_ctx.state = ST_MOVING;
  UNLOCK_STATE();
  if (sup) emit_done(sseq, DONE_SUPERSEDED, sodom);
}

void ctl_move(float dist, float speed, uint32_t seq) {
  bool sup = false; uint32_t sseq = 0; Odom sodom;
  LOCK_STATE();
  sup = begin_finite_locked(sseq, sodom);
  FiniteCmd f;
  f.kind = CMD_MOVE; f.seq = seq;
  f.target_dist = dist;
  f.speed = fabsf(speed);
  g_ctx.finite = f;
  g_ctx.cmd_mode = CMD_MOVE;
  g_ctx.cmd_seq = seq;
  if (g_ctx.state == ST_IDLE) g_ctx.state = ST_MOVING;
  UNLOCK_STATE();
  if (sup) emit_done(sseq, DONE_SUPERSEDED, sodom);
}

void ctl_come(float heading_deg, float stop_at, uint32_t seq) {
  bool sup = false; uint32_t sseq = 0; Odom sodom;
  LOCK_STATE();
  sup = begin_finite_locked(sseq, sodom);
  FiniteCmd f;
  f.kind = CMD_COME; f.seq = seq;
  f.target_dtheta = DEG2RAD(heading_deg);
  f.rate = DEG2RAD(g_ctx.params.default_turn_rate);
  f.speed = g_ctx.params.max_lin;
  f.come_stop_at = stop_at;
  f.come_sim_wall = stop_at + 0.6f;       // stub: advance ~0.6 m then stop
  f.come_turning = (fabsf(heading_deg) > 1.0f);
  g_ctx.finite = f;
  g_ctx.cmd_mode = CMD_COME;
  g_ctx.cmd_seq = seq;
  if (g_ctx.state == ST_IDLE) g_ctx.state = ST_MOVING;
  UNLOCK_STATE();
  if (sup) emit_done(sseq, DONE_SUPERSEDED, sodom);
}

void ctl_wheel_test(int side, float frac, uint32_t ms, uint32_t seq) {
  // Single-wheel bring-up jog: arm CMD_WHEEL for `ms` ms, then control_tick auto-stops
  // it (done:completed). Runs as an AUTO finite command so the Mac/bench heartbeat keeps
  // it alive and a dropped link trips the watchdog -> motors off within watchdog_ms; a
  // gamepad taking over, `stop`, or `estop` supersede/cut it like any finite command.
  bool sup = false; uint32_t sseq = 0; Odom sodom;
  LOCK_STATE();
  sup = begin_finite_locked(sseq, sodom);
  FiniteCmd f;
  f.kind = CMD_WHEEL; f.seq = seq;
  f.wheel_side = (side != 0) ? 1 : 0;
  f.wheel_frac = frac;
  f.wheel_start_ms = millis();
  f.wheel_ms = ms;
  g_ctx.finite = f;
  g_ctx.cmd_mode = CMD_WHEEL;
  g_ctx.cmd_seq = seq;
  if (g_ctx.state == ST_IDLE) g_ctx.state = ST_MOVING;
  UNLOCK_STATE();
  if (sup) emit_done(sseq, DONE_SUPERSEDED, sodom);
}

void ctl_stop(uint32_t seq) {
  bool sup = false; uint32_t sseq = 0; Odom sodom;
  LOCK_STATE();
  if (g_ctx.finite.kind != CMD_NONE && g_ctx.finite.kind != CMD_DRIVE) {
    sup = true; sseq = g_ctx.finite.seq; sodom = g_ctx.odom;
  }
  g_ctx.finite = FiniteCmd();
  g_ctx.setpoint.lin = 0; g_ctx.setpoint.ang = 0;
  g_ctx.cmd_mode = CMD_NONE;
  g_ctx.cmd_seq = seq;
  // state falls to IDLE via control_tick as the plant ramps down.
  UNLOCK_STATE();
  if (sup) emit_done(sseq, DONE_SUPERSEDED, sodom);
}

void ctl_estop(uint32_t seq) {
  bool sup = false; uint32_t sseq = 0; Odom sodom;
  LOCK_STATE();
  if (g_ctx.finite.kind != CMD_NONE && g_ctx.finite.kind != CMD_DRIVE) {
    sup = true; sseq = g_ctx.finite.seq; sodom = g_ctx.odom;
  }
  g_ctx.finite = FiniteCmd();
  g_ctx.setpoint.lin = 0; g_ctx.setpoint.ang = 0;
  g_ctx.cmd_mode = CMD_NONE;
  g_ctx.state = ST_ESTOP;
  g_ctx.cmd_seq = seq;
  UNLOCK_STATE();
  if (sup) emit_done(sseq, DONE_ESTOPPED, sodom);
  emit_event("estop");
}

bool ctl_clear(uint32_t seq) {
  bool cleared = false;
  const char* ev = nullptr;
  LOCK_STATE();
  g_ctx.cmd_seq = seq;
  if (g_ctx.state == ST_ESTOP) {
    g_ctx.state = ST_IDLE; cleared = true; ev = "estop_clear";
  } else if (g_ctx.state == ST_FAULT) {
    g_ctx.state = ST_IDLE; g_ctx.fault = F_NONE; cleared = true; ev = "fault_clear";
  }
  UNLOCK_STATE();
  if (cleared && ev) emit_event(ev);
  return cleared;
}

// ---- Manual (gamepad) control (docs §11) ---------------------------------
// Driven from gamepad.cpp on the ESP32 — entirely independent of the Mac link, so
// it works even with the USB unplugged. owner=MANUAL makes proto_io's motion_gate
// reject Mac drive/turn/move/come; stop/estop/config/ping still pass.

// A live pad input takes over from a dead Mac link (field fix 2026-07-11): when Rex
// shuts down, the heartbeat watchdog latches ST_COMMS_LOST, which control_tick treats
// as HALTED — and the only exit was a valid Mac line, so the paired gamepad went
// unresponsive until the app was relaunched. But comms-lost exists to stop AUTONOMOUS
// motion with nobody at the wheel (the watchdog deliberately only guards OWNER_AUTO,
// safety.cpp) — a human pushing the stick IS someone at the wheel, so every manual
// entry clears the latch and proceeds. estop/fault still require an explicit clear
// (Start button), exactly as before. Caller holds the state lock.
static inline void manual_takeover_clears_comms_lost() {
  if (g_ctx.state == ST_COMMS_LOST) {
    g_ctx.state = ST_IDLE;
    if (g_ctx.fault == F_COMMS_LOST) g_ctx.fault = F_NONE;
  }
}

void ctl_manual_drive(float lin, float ang, float pivot_blend) {
  // Refreshed every gamepad poll; the drive deadman stops the base if polls stall.
  bool sup = false; uint32_t sseq = 0; Odom sodom;
  LOCK_STATE();
  // Don't punch through a hard latch — estop/fault must be cleared first.
  if (g_ctx.state == ST_ESTOP || g_ctx.state == ST_FAULT) { UNLOCK_STATE(); return; }
  manual_takeover_clears_comms_lost();      // a live operator outranks a dead Mac link
  sup = begin_finite_locked(sseq, sodom);   // taking over from an autonomous finite cmd
  g_ctx.owner = OWNER_MANUAL;
  g_ctx.finite = FiniteCmd();
  g_ctx.setpoint.lin = lin; g_ctx.setpoint.ang = ang;
  g_ctx.setpoint.pivot_blend = clampf(pivot_blend, 0.0f, 1.0f);
  g_ctx.cmd_mode = CMD_DRIVE;
  g_ctx.drive_set_ms = millis();
  if (fabsf(lin) > 0.01f || fabsf(ang) > 0.01f) g_ctx.last_manual_input_ms = millis();
  if (g_ctx.state == ST_IDLE) g_ctx.state = ST_MOVING;
  UNLOCK_STATE();
  if (sup) emit_done(sseq, DONE_SUPERSEDED, sodom);
}

void ctl_manual_turn(float deg, float rate_dps) {
  // Gamepad D-pad "spin to a heading" (encoder validation). Same encoder-closed-loop turn
  // as ctl_turn, but armed as a MANUAL finite command: owner becomes MANUAL so (a) the
  // heartbeat watchdog won't abort it — it survives a USB drop like stick teleop — and
  // (b) the Mac can't issue a competing autonomous move. The caller passes a RELATIVE,
  // signed delta in degrees (it computes shortest-path from the live encoder heading).
  // gamepad_tick suppresses its zero-stick manual deadman while CMD_TURN is in flight so
  // this isn't superseded every poll; a real stick push still takes over and cancels it.
  bool sup = false; uint32_t sseq = 0; Odom sodom;
  LOCK_STATE();
  // Don't punch through a hard latch — estop/fault must be cleared first (mirror ctl_manual_drive).
  if (g_ctx.state == ST_ESTOP || g_ctx.state == ST_FAULT) { UNLOCK_STATE(); return; }
  manual_takeover_clears_comms_lost();      // a live operator outranks a dead Mac link
  sup = begin_finite_locked(sseq, sodom);
  FiniteCmd f;
  f.kind = CMD_TURN; f.seq = 0;
  f.target_dtheta = DEG2RAD(deg);
  f.rate = DEG2RAD(rate_dps);
  g_ctx.owner = OWNER_MANUAL;
  g_ctx.finite = f;
  g_ctx.cmd_mode = CMD_TURN;
  g_ctx.last_manual_input_ms = millis();   // a deliberate manual input — defer idle auto-return
  if (g_ctx.state == ST_IDLE) g_ctx.state = ST_MOVING;
  UNLOCK_STATE();
  if (sup) emit_done(sseq, DONE_SUPERSEDED, sodom);
}

void ctl_manual_move(float dist, float speed) {
  // Gamepad D-pad Up/Down "nudge forward/back" — the linear mirror of ctl_manual_turn.
  // Same encoder-closed finite move as ctl_move, but armed as a MANUAL command: owner
  // becomes MANUAL so the heartbeat watchdog won't abort it (survives a USB drop) and
  // the Mac can't issue a competing autonomous move. Unlike a turn (pure spin), a move
  // HAS a travel direction, so the ToF stop reflex gates it like any finite move: driving
  // toward a Z_STOP obstacle terminates it with done:blocked (control_tick §7.4 path).
  // gamepad_tick suppresses its zero-stick manual deadman while a finite nudge is in
  // flight; a real stick push still takes over and cancels it (intended override).
  bool sup = false; uint32_t sseq = 0; Odom sodom;
  LOCK_STATE();
  // Don't punch through a hard latch — estop/fault must be cleared first (mirror ctl_manual_drive).
  if (g_ctx.state == ST_ESTOP || g_ctx.state == ST_FAULT) { UNLOCK_STATE(); return; }
  manual_takeover_clears_comms_lost();      // a live operator outranks a dead Mac link
  sup = begin_finite_locked(sseq, sodom);
  FiniteCmd f;
  f.kind = CMD_MOVE; f.seq = 0;
  f.target_dist = dist;
  f.speed = fabsf(speed);
  g_ctx.owner = OWNER_MANUAL;
  g_ctx.finite = f;
  g_ctx.cmd_mode = CMD_MOVE;
  g_ctx.last_manual_input_ms = millis();   // a deliberate manual input — defer idle auto-return
  if (g_ctx.state == ST_IDLE) g_ctx.state = ST_MOVING;
  UNLOCK_STATE();
  if (sup) emit_done(sseq, DONE_SUPERSEDED, sodom);
}

void ctl_manual_stop() {
  // Disconnect failsafe: stop NOW but keep MANUAL ownership, so AUTO doesn't silently
  // resume on a dropped pad (docs §11.4). Auto-return (if enabled) handles the handoff.
  LOCK_STATE();
  g_ctx.setpoint.lin = 0; g_ctx.setpoint.ang = 0;
  if (g_ctx.cmd_mode == CMD_DRIVE) g_ctx.cmd_mode = CMD_NONE;
  g_ctx.full_override = false;
  UNLOCK_STATE();
}

void ctl_manual_release() {
  // Explicit (Start button) or idle-timeout return of control to AUTO.
  LOCK_STATE();
  g_ctx.setpoint.lin = 0; g_ctx.setpoint.ang = 0;
  if (g_ctx.cmd_mode == CMD_DRIVE) g_ctx.cmd_mode = CMD_NONE;
  g_ctx.full_override = false;
  g_ctx.owner = OWNER_AUTO;
  UNLOCK_STATE();
}

void ctl_set_full_override(bool on) {
  LOCK_STATE(); g_ctx.full_override = on; UNLOCK_STATE();
}

void ctl_set_gamepad(bool connected) {
  LOCK_STATE(); g_ctx.gamepad = connected ? GP_CONNECTED : GP_NONE; UNLOCK_STATE();
}
