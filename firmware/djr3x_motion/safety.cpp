#include "safety.h"
#include "proto_io.h"
#include <math.h>

// NOTE: this radial ToF layout (8 horizontal sensors) has NO down-facing cliff
// sensor, so cliff/stair-edge detection is NOT available — the base will drive off
// a drop-off. Reflex protection here is obstacle-only (front/rear zones).
// FAIL-OPEN: if every sensor in the travel direction reads -1 (all errored / unwired),
// min3_valid yields 32767 and the zone is CLEAR — the base is NOT stopped. tof.cpp
// holds each sensor's last-good value through a transient -1, so this only bites when
// a whole direction is genuinely dead/unwired (acceptable during bring-up; revisit to
// fail-safe — treat persistent all-error as STOP — before trusting autonomy).

// Edge-trigger memory (only this task touches these).
static bool       s_prev_blocked = false;
static MotionDir  s_prev_block_dir = DIR_NONE;
static MotionZone s_prev_zone_for_block = Z_CLEAR;

void safety_init() {
  s_prev_blocked = false;
  s_prev_block_dir = DIR_NONE;
  s_prev_zone_for_block = Z_CLEAR;
}

static int16_t min2_valid(int16_t a, int16_t b) {
  int16_t m = 32767;
  if (a >= 0 && a < m) m = a;
  if (b >= 0 && b < m) m = b;
  return m;
}

void note_mac_line() {
  bool restored = false;
  LOCK_STATE();
  g_ctx.last_mac_ms = millis();
  g_ctx.seen_mac = true;
  if (g_ctx.state == ST_COMMS_LOST) {
    g_ctx.state = ST_IDLE;
    if (g_ctx.fault == F_COMMS_LOST) g_ctx.fault = F_NONE;
    restored = true;
  }
  UNLOCK_STATE();
  if (restored) emit_event("comms_restored");
}

void safety_tick() {
  uint32_t now = millis();

  bool      ev_comms_lost = false;
  bool      ev_block      = false;
  bool      ev_block_cliff = false;
  MotionDir ev_block_dir  = DIR_NONE;
  bool      ev_aborted    = false;   // finite cmd aborted by comms loss
  uint32_t  ab_seq        = 0;
  Odom      ab_odom;

  LOCK_STATE();
  MotionContext& c = g_ctx;

  // ---- Heartbeat watchdog (estop outranks comms_lost) ----
  // Only guards AUTONOMOUS motion: while a gamepad owns the base (OWNER_MANUAL) the
  // Mac link is irrelevant — manual control must survive a USB unplug (docs §11). The
  // gamepad's own disconnect failsafe covers liveness there.
  if (c.seen_mac && c.owner == OWNER_AUTO && c.state != ST_ESTOP && c.state != ST_COMMS_LOST &&
      (uint32_t)(now - c.last_mac_ms) > c.params.watchdog_ms) {
    c.state = ST_COMMS_LOST;
    c.fault = F_COMMS_LOST;
    c.setpoint.lin = 0; c.setpoint.ang = 0;
    c.cmd_mode = CMD_NONE;
    // Abort any in-flight finite cmd ATOMICALLY here (clear under the same lock
    // that enters COMMS_LOST), so a recovering Mac line that flips us back to
    // IDLE can't resume a stale finite without its done:aborted having been sent.
    if (c.finite.kind != CMD_NONE) {
      ev_aborted = true; ab_seq = c.finite.seq; ab_odom = c.odom;
      c.finite = FiniteCmd();
    }
    ev_comms_lost = true;
  }

  // ---- Zone evaluation — BOTH directions, ALWAYS ON (owner request 2026-07-16:
  // avoidance must never sleep; a parked base must refuse to drive into a wall
  // under any circumstances). Front (fl+fr) and rear (rl+rr) evaluate every tick,
  // independent of motion or command intent; each direction blocks independently
  // (control_tick zeroes only the blocked component — turning and driving AWAY
  // from a block are always free). The side SHORT pairs are deliberately NOT in
  // the reflex: they point 67.5° off the travel axis, so a parallel hallway wall
  // ~250 mm away reads ~270 mm on them and would pin the base in SLOW forever —
  // hallway wall handling belongs to the steering assist (control.cpp).
  //
  // Speed-adaptive envelope (context.h helpers): the configured zones apply at full
  // speed, shrinking to the calib.h floors at rest — fast approach brakes early,
  // slow positioning gets close, and the STOP floor still makes contact impossible.
  const int stop_mm = (int)(stop_zone_eff(c.params, c.odom.lin) * 1000.0f);
  const int slow_mm = (int)(slow_zone_eff(c.params, c.odom.lin) * 1000.0f);
  const int16_t d_front = min2_valid(c.tof.fl, c.tof.fr);
  const int16_t d_rear  = min2_valid(c.tof.rl, c.tof.rr);
  const MotionZone zf = (d_front < stop_mm) ? Z_STOP : ((d_front < slow_mm) ? Z_SLOW : Z_CLEAR);
  const MotionZone zr = (d_rear  < stop_mm) ? Z_STOP : ((d_rear  < slow_mm) ? Z_SLOW : Z_CLEAR);
  const MotionZone z  = (zf >= zr) ? zf : zr;   // worst-of, for state/telemetry/rumble

  c.zone = z;
  MotionDir stop_dir = DIR_NONE;
  if      (zf == Z_STOP && zr == Z_STOP) stop_dir = DIR_BOTH;
  else if (zf == Z_STOP)                 stop_dir = DIR_FRONT;
  else if (zr == Z_STOP)                 stop_dir = DIR_REAR;

  // Outside BLOCKED, blocked_dir mirrors what would trip the reflex this tick.
  // Once BLOCKED, however, the direction is a LATCH: braking makes measured speed
  // fall, which shrinks the speed-adaptive STOP envelope and can reclassify the
  // exact same obstacle as SLOW. Clearing blocked_dir at that point let a still-held
  // gamepad stick accelerate again, expand the envelope, and immediately re-block
  // (violent brake/go oscillation on the tall robot).
  if (c.state != ST_BLOCKED) c.blocked_dir = stop_dir;

  // ---- Reflex stop: only toggles within the IDLE/MOVING/BLOCKED group ----
  // A gamepad operator holding full-override (docs §11.4) deliberately bypasses the
  // zone/cliff reflex — don't enter (and release any) BLOCKED while it's held. The
  // zone is still reported in telemetry so the operator sees what they're overriding.
  if ((c.state == ST_IDLE || c.state == ST_MOVING) && !c.full_override) {
    if (z == Z_STOP || z == Z_CLIFF) {
      c.state = ST_BLOCKED;
      c.blocked_dir = stop_dir;
    }
  } else if (c.state == ST_BLOCKED) {
    if (c.full_override) {
      c.state = ST_IDLE;
      c.blocked_dir = DIR_NONE;
    } else {
      // Add a newly blocked direction while preserving the original latch. This
      // matters if backing away from a front obstacle discovers a rear obstacle.
      const bool latched_front =
          c.blocked_dir == DIR_FRONT || c.blocked_dir == DIR_BOTH;
      const bool latched_rear =
          c.blocked_dir == DIR_REAR || c.blocked_dir == DIR_BOTH;
      bool keep_front = latched_front && zf != Z_CLEAR;
      bool keep_rear  = latched_rear  && zr != Z_CLEAR;
      if (zf == Z_STOP) keep_front = true;
      if (zr == Z_STOP) keep_rear = true;

      if      (keep_front && keep_rear) c.blocked_dir = DIR_BOTH;
      else if (keep_front)              c.blocked_dir = DIR_FRONT;
      else if (keep_rear)               c.blocked_dir = DIR_REAR;
      else {
        // A latched direction releases only after that direction reaches CLEAR.
        // SLOW is intentionally not enough: the stick may still be held toward the
        // obstacle, and the progressive taper resumes safely after genuine clearance.
        c.state = ST_IDLE;
        c.blocked_dir = DIR_NONE;
      }
    }
  }

  // Edge-detect a new block for the event stream.
  bool now_blocked = (c.state == ST_BLOCKED);
  if (now_blocked && (!s_prev_blocked || s_prev_zone_for_block != z || s_prev_block_dir != c.blocked_dir)) {
    ev_block = true;
    ev_block_cliff = (z == Z_CLIFF);
    ev_block_dir = c.blocked_dir;
  }
  s_prev_blocked = now_blocked;
  s_prev_block_dir = c.blocked_dir;
  s_prev_zone_for_block = z;

  UNLOCK_STATE();

  // ---- Emit events outside the lock ----
  if (ev_aborted) emit_done(ab_seq, DONE_ABORTED, ab_odom);
  if (ev_comms_lost) emit_event("comms_lost");
  if (ev_block) {
    if (ev_block_cliff) emit_event("cliff");
    else                emit_event_kv("zone_block", "blocked_dir", dir_str(ev_block_dir));
  }
}
