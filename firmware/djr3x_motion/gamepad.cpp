// gamepad.cpp — Bluetooth gamepad manual override via Bluepad32 (docs §11).
// Built only when MOTION_GAMEPAD_PRESENT==1 (which needs the Bluepad32 board package —
// see README). Otherwise both hooks are no-ops and there's no BT dependency.
#include "gamepad.h"

#if MOTION_GAMEPAD_PRESENT
// ===========================================================================
// REAL — Bluepad32. Target pad: 8BitDo Pro 2 (any Bluepad32-supported pad works;
// the library normalizes them, so this mapping is pad-agnostic). Pairing/mode notes
// in README. Mapping (docs §11.2):
//   left stick : arcade drive — Y forward/back, X turn (avoidance ACTIVE)
//   D-pad      : relative nudges — Up/Down = short fwd/back move, Left/Right = 90° turn
//                (hold L1 + D-pad = absolute-heading encoder test, bring-up only)
//   L3 (click) : toggle surface mode (hardwood <-> carpet)
//   B          : E-STOP (always honored)
//   Start      : clear e-stop + return control to AUTO
//   R3 (click) : toggle SENSOR-BYPASS mode (owner spec 2026-07-16). While ON, the
//                RIGHT stick drives with ALL ToF gating off (for escaping a stuck
//                block) and the left stick is ignored; click R3 again to exit.
//                Auto-cancels on disconnect. Replaces the old hold-L2+R2 override.
// ===========================================================================
#include <Bluepad32.h>
#include <Arduino.h>
#include <math.h>
#include "context.h"
#include "control.h"
#include "calib.h"
#include "proto_io.h"   // emit_event_kv — forward action-button presses to the Mac

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif
// Shortest-path angle wrap to (-pi, pi] + a deg->rad helper, for the D-pad heading turns.
static inline float gp_wrap_pi(float a) {
  while (a >   (float)M_PI) a -= 2.0f * (float)M_PI;
  while (a <= -(float)M_PI) a += 2.0f * (float)M_PI;
  return a;
}
static inline float gp_deg2rad(float d) { return d * (float)M_PI / 180.0f; }

static ControllerPtr s_ctl = nullptr;     // the one pad we drive from
static bool s_prev_b = false;
static bool s_prev_start = false;
static bool s_bypass = false;             // R3-toggled sensor-bypass (right-stick drive)
static uint8_t s_prev_dpad = 0;           // D-pad rising-edge state (heading-turn triggers)
static bool s_prev_l3 = false;            // left-stick-click rising edge (surface-mode toggle)
static bool s_prev_r3 = false;            // right-stick-click rising edge (sensor-bypass toggle)
// SURFACE MODE (replaced the 3 abstract speed levels 2026-07-11 — at full build
// weight the whole ladder was insufficient; what actually varies is the floor):
// false = HARDWOOD (boot/reconnect default), true = CARPET (max authority). L3
// toggles. Each mode sets its own lin/ang ceilings AND the pivot breakaway kick
// (g_ctx.spin_breakaway_duty — a surface property, so autonomous turns get it too).
static bool s_carpet_mode = false;

static void apply_surface_mode() {
  LOCK_STATE();
  g_ctx.spin_breakaway_duty =
      s_carpet_mode ? GAMEPAD_CARPET_SPIN_KICK : GAMEPAD_HARDWOOD_SPIN_KICK;
  g_ctx.spin_run_duty =
      s_carpet_mode ? GAMEPAD_CARPET_SPIN_RUN : GAMEPAD_HARDWOOD_SPIN_RUN;
  UNLOCK_STATE();
}

static void onConnect(ControllerPtr c) {
  if (!s_ctl) {
    s_ctl = c;
    ctl_set_gamepad(true);   // take the first pad; filter reads in tick
    // Observability (field lesson 2026-07-20: pad state was invisible in every
    // log, so a no-pair bug took raw-serial archaeology). Callbacks run inside
    // BP32.update() on the loopTask, so emitting here is as safe as in tick.
    emit_event_kv("gamepad", "state", "connected");
  } else {
    emit_event_kv("gamepad", "state", "ignored_second_pad");
  }
}

static void onDisconnect(ControllerPtr c) {
  if (s_ctl == c) {
    s_ctl = nullptr;
    s_bypass = false;                 // bypass NEVER survives a disconnect
    ctl_set_full_override(false);
    s_carpet_mode = false;    // a reconnected pad starts in HARDWOOD (gentler default)
    apply_surface_mode();
    ctl_set_gamepad(false);
    LOCK_STATE(); g_ctx.gp_live.connected = false; UNLOCK_STATE();  // GUI: pad gone
    ctl_manual_stop();        // failsafe: stop now, KEEP manual — never silently resume AUTO
    emit_event_kv("gamepad", "state", "disconnected");
  }
}

// Stick raw (~-512..511) -> -1..1 with a center deadzone, rescaled so motion starts
// smoothly at the deadzone edge instead of jumping.
static float stick_norm(int v) {
  float n = (float)v / 512.0f;
  if (n > 1.0f) n = 1.0f; else if (n < -1.0f) n = -1.0f;
  float a = fabsf(n);
  if (a < GAMEPAD_DEADZONE) return 0.0f;
  float s = (a - GAMEPAD_DEADZONE) / (1.0f - GAMEPAD_DEADZONE);
  return (n < 0.0f) ? -s : s;
}

// Idle-timeout return to AUTO (only if MOTION_MANUAL_AUTORETURN). Runs every tick,
// including while disconnected — so a dropped pad eventually hands back to AUTO if
// auto-return is enabled, and stays MANUAL+stopped if it isn't (docs §11.3/§11.4).
static void maybe_autoreturn() {
  bool manual, autoreturn; uint32_t idle_ms, last_in, now = millis();
  LOCK_STATE();
  manual     = (g_ctx.owner == OWNER_MANUAL);
  autoreturn = g_ctx.params.manual_autoreturn;
  idle_ms    = g_ctx.params.manual_idle_return_secs * 1000u;
  last_in    = g_ctx.last_manual_input_ms;
  UNLOCK_STATE();
  if (manual && autoreturn && (uint32_t)(now - last_in) > idle_ms) ctl_manual_release();
}

// ---------------------------------------------------------------------------
// Action buttons -> Mac. The buttons MOTION does NOT use (B=estop, Start, L1/R1,
// L2+R2, sticks are taken) are forwarded as `event:"button"` so R3X can trigger
// sound clips / servo animations on the Mac. Rising-edge only (one event per press),
// emitted whenever the pad is connected — INDEPENDENT of drive owner, so the
// soundboard works in AUTO too and pressing them does NOT grab the wheel.
// btn names must match config.MOTION_GAMEPAD_BUTTON_ACTIONS keys on the Mac.
// NOTE: the D-pad is intentionally NOT forwarded here — it drives the relative motion
// nudges (and, with L1 held, the absolute-heading encoder test) in gamepad_tick below.
// ---------------------------------------------------------------------------
static uint16_t s_prev_actions = 0;

static void poll_action_buttons(ControllerPtr c) {
  struct ActionBtn { const char* name; bool pressed; };
  const ActionBtn btns[] = {
    {"a",          c->a()},
    {"x",          c->x()},
    {"y",          c->y()},
    {"select",     c->miscSelect()},   // the "-" button
    {"home",       c->miscSystem()},   // the star / home button
    // NB: L3 and R3 (stick clicks) are NOT forwarded — L3 toggles the surface
    // mode and R3 toggles sensor-bypass (see gamepad_tick), so they must not
    // also fire soundboard clips.
  };
  const uint8_t n = (uint8_t)(sizeof(btns) / sizeof(btns[0]));
  uint16_t cur = 0;
  for (uint8_t i = 0; i < n; i++) {
    if (btns[i].pressed) cur |= (uint16_t)(1u << i);
    if (btns[i].pressed && !(s_prev_actions & (uint16_t)(1u << i))) {
      emit_event_kv("button", "btn", btns[i].name);   // rising edge
    }
  }
  s_prev_actions = cur;
}

// ---------------------------------------------------------------------------
// Rumble (force feedback) — tactile echo of the collision avoidance + a greeting
// when a host connects. ALL Bluepad32 calls stay on the loopTask (this file's
// tick); other tasks only set the pending flag below. Simple pulse scheduler:
// rumble_burst() queues N pulses, rumble_service() plays them out per tick.
// ---------------------------------------------------------------------------
#if GAMEPAD_RUMBLE_ENABLED
static uint8_t  s_rum_left = 0;             // pulses remaining in the current burst
static uint32_t s_rum_next_ms = 0;          // when the next pulse may fire
static uint16_t s_rum_dur = 0, s_rum_gap = 0;
static uint8_t  s_rum_weak = 0, s_rum_strong = 0;
static bool     s_prev_rum_blocked = false; // edge detect: BLOCKED thump
static bool     s_prev_rum_slow = false;    // edge detect: braking-band buzz
static uint32_t s_last_block_rum_ms = 0;    // re-thump cadence while pushing into a block

static void rumble_burst(uint8_t pulses, uint16_t dur_ms, uint16_t gap_ms,
                         uint8_t weak, uint8_t strong) {
  s_rum_left = pulses; s_rum_dur = dur_ms; s_rum_gap = gap_ms;
  s_rum_weak = weak; s_rum_strong = strong;
  s_rum_next_ms = millis();                 // first pulse fires this tick
}

static void rumble_service(ControllerPtr c, uint32_t now) {
  if (!s_rum_left || (int32_t)(now - s_rum_next_ms) < 0) return;
  c->playDualRumble(0, s_rum_dur, s_rum_weak, s_rum_strong);
  s_rum_next_ms = now + (uint32_t)s_rum_dur + s_rum_gap;
  s_rum_left--;
}
#endif  // GAMEPAD_RUMBLE_ENABLED

// Set from the SERIAL task when a `hello` command arrives (main.py connecting);
// consumed on the loopTask. volatile is enough — a lost race costs one greeting.
static volatile bool     s_hello_rum_pending = false;
static volatile uint32_t s_hello_rum_at_ms = 0;

void gamepad_notify_host_connected() {
  s_hello_rum_at_ms = millis();
  s_hello_rum_pending = true;
}

void gamepad_init() {
  BP32.setup(&onConnect, &onDisconnect);
  BP32.enableVirtualDevice(false);          // real gamepads only (no virtual mouse/kbd)
  BP32.enableNewBluetoothConnections(true); // accept a pad in pairing mode
  apply_surface_mode();                     // boot in the HARDWOOD profile (incl. spin kick)
  emit_log("info", "gamepad: Bluepad32 ready - accepting pad connections");
}

void gamepad_tick() {
  BP32.update();
  ControllerPtr c = s_ctl;
  if (!c || !c->isConnected() || !c->isGamepad()) {
    LOCK_STATE(); g_ctx.gp_live.connected = false; UNLOCK_STATE();  // GUI: no pad
    maybe_autoreturn();
    return;
  }

  // B = E-STOP (rising edge; always honored, even mid-override).
  bool b = c->b();
  if (b && !s_prev_b) ctl_estop(0);
  s_prev_b = b;

  // Start = clear any e-stop/fault + hand control back to AUTO (rising edge).
  bool start = c->miscStart();
  if (start && !s_prev_start) { ctl_clear(0); ctl_manual_release(); }
  s_prev_start = start;

  // Left-stick CLICK (L3) = toggle the SURFACE MODE: hardwood <-> carpet (rising
  // edge). Latches. Rumble confirms without looking: 1 pulse = hardwood, 2 = carpet.
  bool l3 = c->thumbL();
  if (l3 && !s_prev_l3) {
    s_carpet_mode = !s_carpet_mode;
    apply_surface_mode();
    emit_event_kv("mode", "surface", s_carpet_mode ? "carpet" : "hardwood");
#if GAMEPAD_RUMBLE_ENABLED
    rumble_burst(s_carpet_mode ? 2 : 1, 140, 140,
                 GAMEPAD_RUMBLE_HELLO_MAG, GAMEPAD_RUMBLE_HELLO_MAG);
#endif
  }
  s_prev_l3 = l3;

  // D-pad -> RELATIVE driving nudges (rising edge: one action per press):
  //   Up    = nudge FORWARD  by GAMEPAD_NUDGE_DIST_M (finite move — ToF stop reflex gates it)
  //   Down  = nudge BACKWARD by GAMEPAD_NUDGE_DIST_M (finite move — ToF-gated, rear pair)
  //   Left  = turn LEFT  (CCW) by params.default_turn_deg (90°), relative to HERE
  //   Right = turn RIGHT (CW)  by params.default_turn_deg, relative to HERE
  // All run as MANUAL finite commands: the heartbeat watchdog won't abort them (they
  // survive a USB drop like stick teleop) and the Mac can't fight them; a left-stick
  // push cancels an in-flight nudge. Turns are pure spins (lin≈0), so ToF does NOT
  // gate them — mind the ring swing near obstacles.
  //
  // Hold L1 + D-pad = the original ABSOLUTE-heading encoder test (Up=0 Left=+90 Down=180
  // Right=-90, REP-103 body frame, shortest-path from the live odom heading). Use it to
  // validate encoder signs + counts/track calibration: a flipped sign runs away, a wrong
  // counts_per_meter/track_width_m scale over/under-rotates (docs §14). Bring-up only.
  {
    const uint8_t dp = c->dpad();   // bitmask: UP=0x01 DOWN=0x02 RIGHT=0x04 LEFT=0x08
    const uint8_t rising = (uint8_t)(dp & (uint8_t)~s_prev_dpad);
    s_prev_dpad = dp;
    if (rising) {
      float theta, rate, turn_deg, nudge_speed;
      LOCK_STATE();
      theta       = g_ctx.odom.theta;
      rate        = g_ctx.params.default_turn_rate;
      turn_deg    = g_ctx.params.default_turn_deg;
      nudge_speed = g_ctx.params.max_lin * GAMEPAD_NUDGE_SPEED_FRAC;
      UNLOCK_STATE();
      if (c->l1()) {
        // Absolute-heading encoder test (headings in the REP-103 body frame, +deg = CCW).
        struct DpadTurn { uint8_t bit; float heading_deg; };
        static const DpadTurn DPAD_TURNS[] = {
          {0x01,   0.0f},   // Up    -> 0
          {0x08,  90.0f},   // Left  -> +90 (CCW)
          {0x02, 180.0f},   // Down  -> 180
          {0x04, -90.0f},   // Right -> -90 (CW)
        };
        for (uint8_t i = 0; i < (uint8_t)(sizeof(DPAD_TURNS) / sizeof(DPAD_TURNS[0])); i++) {
          if (rising & DPAD_TURNS[i].bit) {
            const float delta_deg =
                gp_wrap_pi(gp_deg2rad(DPAD_TURNS[i].heading_deg) - theta) * 180.0f / (float)M_PI;
            ctl_manual_turn(delta_deg, rate);
            break;                                // one action per press (diagonal picks first)
          }
        }
      } else if (rising & 0x01) {                 // Up: nudge forward
        ctl_manual_move(+GAMEPAD_NUDGE_DIST_M, nudge_speed);
      } else if (rising & 0x02) {                 // Down: nudge backward
        ctl_manual_move(-GAMEPAD_NUDGE_DIST_M, nudge_speed);
      } else if (rising & 0x08) {                 // Left: relative turn CCW
        ctl_manual_turn(+turn_deg, rate);
      } else if (rising & 0x04) {                 // Right: relative turn CW
        ctl_manual_turn(-turn_deg, rate);
      }
    }
  }

  // SENSOR-BYPASS toggle (R3 rising edge). While ON: full_override stays asserted
  // (reflex block, slow taper, and hallway assist all stand down) and the drive
  // command comes from the RIGHT stick only — a deliberately different hand
  // position, so normal left-stick driving can never accidentally run unprotected.
  // Rumble confirms without looking: 3 pulses = bypass ON, 1 pulse = back to normal.
  bool r3 = c->thumbR();
  if (r3 && !s_prev_r3) {
    s_bypass = !s_bypass;
    ctl_set_full_override(s_bypass);
    emit_event_kv("mode", "bypass", s_bypass ? "on" : "off");
#if GAMEPAD_RUMBLE_ENABLED
    rumble_burst(s_bypass ? 3 : 1, 140, 140,
                 GAMEPAD_RUMBLE_HELLO_MAG, GAMEPAD_RUMBLE_HELLO_MAG);
#endif
  }
  s_prev_r3 = r3;

  // Triggers still read for the GUI button mirror below (their override role is gone).
  float br = (float)c->brake()    / GAMEPAD_TRIGGER_MAX;
  float th = (float)c->throttle() / GAMEPAD_TRIGGER_MAX;

  // Drive stick: LEFT in normal operation (avoidance active), RIGHT in bypass —
  // scaled by the SURFACE MODE's ceilings (full stick = the mode's max).
  float fwd, turn;
  if (s_bypass) {
    fwd  = -stick_norm(c->axisRY());  // right stick up = forward, sensors bypassed
    turn =  stick_norm(c->axisRX());
  } else {
    fwd  = -stick_norm(c->axisY());   // left stick up = forward
    turn =  stick_norm(c->axisX());
  }
  const float mode_lin = s_carpet_mode ? GAMEPAD_CARPET_LIN_MS   : GAMEPAD_HARDWOOD_LIN_MS;
  const float mode_ang = s_carpet_mode ? GAMEPAD_CARPET_ANG_RADS : GAMEPAD_HARDWOOD_ANG_RADS;

  // Teleop scales against the GAMEPAD's OWN ceilings, NOT params.max_lin/max_ang —
  // those are the autonomous caps, and the Mac pushes them down on connect
  // (0.25 m/s), which used to silently slow the pad whenever Rex was running.
  // control_tick clamps manual drives to the carpet (larger) profile defensively.
  // Concave response curve on the forward/back axis (GAMEPAD_LIN_GAMMA < 1): lifts the
  // command at small stick pushes so the loaded base breaks stiction and creeps reliably,
  // while full deflection still lands exactly on the mode max. Turn stays linear, and
  // the spin↔arcade blend below deliberately keys off the RAW `fwd` (its LO/HI bands were
  // tuned on the raw stick fraction — shaping it would shift the blend feel).
  float fwd_shaped = powf(fabsf(fwd), GAMEPAD_LIN_GAMMA);
  if (fwd < 0.0f) fwd_shaped = -fwd_shaped;
  float lin =  fwd_shaped * mode_lin;
  // Spin↔arcade BLEND, keyed off how far the stick is pushed forward/back. bt only
  // drives the wheel MIXING morph in hal now (pure spin -> arcade arc); turn authority
  // is the mode's full ceiling at every blend (the old per-level authority taper died
  // with the speed levels — GAMEPAD_SPIN_SCALE governed it and is retired).
  float bt = clampf((fabsf(fwd) - GAMEPAD_SPIN_BLEND_FWD_LO) /
                    (GAMEPAD_SPIN_BLEND_FWD_HI - GAMEPAD_SPIN_BLEND_FWD_LO), 0.0f, 1.0f);
  bt = bt * bt * (3.0f - 2.0f * bt);       // smoothstep: zero slope at both edges
  float ang = -turn * mode_ang;            // stick-right => -ang (REP-103: +ang = left)

  // Enter MANUAL on the first meaningful push; once manual, keep refreshing (incl. zero,
  // which feeds the drive deadman and holds the base stopped) until release/auto-return.
  // EXCEPTION: while a D-pad nudge is in flight (finite CMD_TURN or CMD_MOVE), skip the
  // zero-stick deadman refresh — ctl_manual_drive would wipe the finite command every poll.
  // A real stick push (meaningful) still takes over and cancels it (intended override).
  bool meaningful = (fabsf(lin) > 0.001f || fabsf(ang) > 0.001f);
  bool isManual, nudgeInFlight;
  LOCK_STATE();
  isManual      = (g_ctx.owner == OWNER_MANUAL);
  nudgeInFlight = (g_ctx.finite.kind == CMD_TURN || g_ctx.finite.kind == CMD_MOVE);
  UNLOCK_STATE();
  if (meaningful || (isManual && !nudgeInFlight)) ctl_manual_drive(lin, ang, bt);

#if GAMEPAD_RUMBLE_ENABLED
  // ---- Rumble feedback ----
  {
    const uint32_t now = millis();
    // Host connect (main.py handshake): friendly double pulse. TTL guards against a
    // stale greet buzzing a pad that's powered on minutes later.
    if (s_hello_rum_pending) {
      s_hello_rum_pending = false;
      if ((uint32_t)(now - s_hello_rum_at_ms) <= GAMEPAD_RUMBLE_HELLO_TTL_MS)
        rumble_burst(2, GAMEPAD_RUMBLE_HELLO_MS, GAMEPAD_RUMBLE_HELLO_GAP_MS,
                     GAMEPAD_RUMBLE_HELLO_MAG, GAMEPAD_RUMBLE_HELLO_MAG);
    }
    bool blocked; MotionZone zn;
    LOCK_STATE(); blocked = (g_ctx.state == ST_BLOCKED); zn = g_ctx.zone; UNLOCK_STATE();
    // Hard stop (BLOCKED): strong thump on entry; re-thump on a slow cadence while the
    // operator keeps pushing into the block, so the pad keeps saying "wall".
    if (blocked && !s_prev_rum_blocked) {
      rumble_burst(1, GAMEPAD_RUMBLE_BLOCK_MS, 0,
                   GAMEPAD_RUMBLE_BLOCK_WEAK, GAMEPAD_RUMBLE_BLOCK_STRONG);
      s_last_block_rum_ms = now;
    } else if (blocked && meaningful &&
               (uint32_t)(now - s_last_block_rum_ms) >= GAMEPAD_RUMBLE_BLOCK_REPEAT_MS) {
      rumble_burst(1, GAMEPAD_RUMBLE_BLOCK_MS / 2, 0,
                   GAMEPAD_RUMBLE_BLOCK_WEAK, GAMEPAD_RUMBLE_BLOCK_STRONG);
      s_last_block_rum_ms = now;
    }
    s_prev_rum_blocked = blocked;
    // Braking band (Z_SLOW, collision avoidance actively slowing the drive): one light
    // buzz on entry — the zone only leaves CLEAR while moving/commanding toward an
    // obstacle, so this is exactly "the taper just grabbed the throttle".
    const bool slow_now = (!blocked && zn == Z_SLOW);
    if (slow_now && !s_prev_rum_slow)
      rumble_burst(1, GAMEPAD_RUMBLE_SLOW_MS, 0,
                   GAMEPAD_RUMBLE_SLOW_WEAK, GAMEPAD_RUMBLE_SLOW_STRONG);
    s_prev_rum_slow = slow_now;

    rumble_service(c, now);
  }
#endif  // GAMEPAD_RUMBLE_ENABLED

  // Forward the soundboard / animation buttons to the Mac (does not affect drive).
  poll_action_buttons(c);

  // Mirror the live pad (stick + ALL buttons) to telemetry for the GUI Motivator
  // Control "physical controller" display. Level state (not edges); reuses turn/fwd/
  // br/th already read this tick. Bit order GP_BTN_* below MUST match the GUI's
  // _GP_BTN_LABELS in gui/dashboard.py.
  //   0 A   1 B   2 X   3 Y   4 L1  5 R1  6 L2  7 R2
  //   8 Up  9 Down 10 Left 11 Right 12 Select 13 Start 14 Home 15 L3 16 R3
  const uint8_t dp = c->dpad();
  uint32_t bm = 0;
  if (c->a())          bm |= (1u << 0);
  if (c->b())          bm |= (1u << 1);
  if (c->x())          bm |= (1u << 2);
  if (c->y())          bm |= (1u << 3);
  if (c->l1())         bm |= (1u << 4);
  if (c->r1())         bm |= (1u << 5);
  if (br >= GAMEPAD_TRIGGER_PRESS_FRAC) bm |= (1u << 6);   // L2 (brake trigger)
  if (th >= GAMEPAD_TRIGGER_PRESS_FRAC) bm |= (1u << 7);   // R2 (throttle trigger)
  if (dp & 0x01)       bm |= (1u << 8);    // up
  if (dp & 0x02)       bm |= (1u << 9);    // down
  if (dp & 0x08)       bm |= (1u << 10);   // left
  if (dp & 0x04)       bm |= (1u << 11);   // right
  if (c->miscSelect()) bm |= (1u << 12);
  if (c->miscStart())  bm |= (1u << 13);
  if (c->miscSystem()) bm |= (1u << 14);
  if (c->thumbL())     bm |= (1u << 15);
  if (c->thumbR())     bm |= (1u << 16);
  LOCK_STATE();
  g_ctx.gp_live.connected = true;
  g_ctx.gp_live.lx = turn;     // right = +
  g_ctx.gp_live.ly = fwd;      // stick-up = +
  g_ctx.gp_live.btn_mask = bm;
  g_ctx.gp_live.batt = c->battery();   // raw 0..255; Mac maps (see proto note)
  UNLOCK_STATE();

  maybe_autoreturn();
}

#else
// ===========================================================================
// STUB — gamepad feature off. No Bluepad32 dependency; all hooks do nothing.
// ===========================================================================
void gamepad_init() {}
void gamepad_tick() {}
void gamepad_notify_host_connected() {}
#endif  // MOTION_GAMEPAD_PRESENT
