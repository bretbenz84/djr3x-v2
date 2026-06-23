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

void control_init() {
  // Nothing yet; plant state lives in g_ctx (zeroed at construction).
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

  // Reflex gating: if blocked in the travel direction, zero that component — UNLESS
  // a gamepad operator is holding full-override (docs §11.4), which deliberately
  // bypasses ToF for nudging through tight spots. estop/fault/comms-loss halts above
  // are NOT bypassed by full_override.
  if (c.state == ST_BLOCKED && !c.full_override) {
    if (c.blocked_dir == DIR_FRONT && lin_t > 0) lin_t = 0;
    if (c.blocked_dir == DIR_REAR  && lin_t < 0) lin_t = 0;
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
  if (halted) hal_motors_off();
  else        hal_drive_velocity(lin_t, ang_t, dt);
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

void ctl_manual_drive(float lin, float ang) {
  // Refreshed every gamepad poll; the drive deadman stops the base if polls stall.
  bool sup = false; uint32_t sseq = 0; Odom sodom;
  LOCK_STATE();
  // Don't punch through a hard latch — estop/fault must be cleared first.
  if (g_ctx.state == ST_ESTOP || g_ctx.state == ST_FAULT) { UNLOCK_STATE(); return; }
  sup = begin_finite_locked(sseq, sodom);   // taking over from an autonomous finite cmd
  g_ctx.owner = OWNER_MANUAL;
  g_ctx.finite = FiniteCmd();
  g_ctx.setpoint.lin = lin; g_ctx.setpoint.ang = ang;
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
