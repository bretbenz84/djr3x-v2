// gamepad.cpp — Bluetooth gamepad manual override via Bluepad32 (docs §11).
// Built only when MOTION_GAMEPAD_PRESENT==1 (which needs the Bluepad32 board package —
// see README). Otherwise both hooks are no-ops and there's no BT dependency.
#include "gamepad.h"

#if MOTION_GAMEPAD_PRESENT
// ===========================================================================
// REAL — Bluepad32. Target pad: 8BitDo Pro 2 (any Bluepad32-supported pad works;
// the library normalizes them, so this mapping is pad-agnostic). Pairing/mode notes
// in README. Mapping (docs §11.2):
//   left stick : arcade drive — Y forward/back, X turn
//   L3 (click) : cycle speed level — slow (default) / faster / full
//   B          : E-STOP (always honored)
//   Start      : clear e-stop + return control to AUTO
//   L2 + R2 (both, near full): hold to FULL-OVERRIDE — bypass ToF gating
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
static bool s_full_override = false;
static uint8_t s_prev_dpad = 0;           // D-pad rising-edge state (heading-turn triggers)
static bool s_prev_l3 = false;            // left-stick-click rising edge (speed-level toggle)
static uint8_t s_speed_level = 0;         // 0 slow (default) / 1 med / 2 full; L3 cycles

// The three teleop speed levels L3 cycles through (fraction of the caps).
static const float SPEED_LEVELS[3] = {
  GAMEPAD_SPEED_SLOW, GAMEPAD_SPEED_MED, GAMEPAD_SPEED_FULL,
};

static void onConnect(ControllerPtr c) {
  if (!s_ctl) { s_ctl = c; ctl_set_gamepad(true); }   // take the first pad; filter reads in tick
}

static void onDisconnect(ControllerPtr c) {
  if (s_ctl == c) {
    s_ctl = nullptr;
    s_full_override = false;
    s_speed_level = 0;        // a reconnected pad starts at the SLOW level, not wherever it left off
    ctl_set_gamepad(false);
    LOCK_STATE(); g_ctx.gp_live.connected = false; UNLOCK_STATE();  // GUI: pad gone
    ctl_manual_stop();        // failsafe: stop now, KEEP manual — never silently resume AUTO
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
// NOTE: the D-pad is intentionally NOT forwarded here — it is repurposed in gamepad_tick
// to spin the base to absolute headings for the encoder-validation test (see below).
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
    {"r3",         c->thumbR()},        // right stick click
    // NB: L3 (left stick click) is NOT forwarded — it cycles the drive speed level
    // (see gamepad_tick), so it must not also fire a soundboard clip.
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

void gamepad_init() {
  BP32.setup(&onConnect, &onDisconnect);
  BP32.enableVirtualDevice(false);          // real gamepads only (no virtual mouse/kbd)
  BP32.enableNewBluetoothConnections(true); // accept a pad in pairing mode
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

  // Left-stick CLICK (L3) = cycle the drive speed level: slow -> faster -> full -> slow
  // (rising edge, one step per press). Latches — it's a mode, not a held modifier.
  bool l3 = c->thumbL();
  if (l3 && !s_prev_l3) {
    s_speed_level = (uint8_t)((s_speed_level + 1) % 3);
    const char lv[2] = { (char)('1' + s_speed_level), '\0' };  // "1"/"2"/"3" for a Mac cue
    emit_event_kv("speed", "level", lv);
  }
  s_prev_l3 = l3;

  // D-pad -> spin the base to an ABSOLUTE heading (encoder validation). Rising edge: one
  // turn per press. Headings in the REP-103 body frame (+deg = CCW / left):
  //   Up = 0   Left = +90 (CCW)   Down = 180   Right = -90 (CW)
  // Each press issues a MANUAL finite turn BY the shortest-path delta from the live encoder
  // heading (g_ctx.odom.theta), so a correctly wired + calibrated base lands square at 90°
  // steps; a flipped encoder sign runs away, a wrong counts/track scale over/under-rotates.
  // It runs as a MANUAL turn (ctl_manual_turn) so the heartbeat watchdog won't abort it and
  // the Mac can't fight it; a left-stick push cancels it. A turn is a pure spin (lin≈0), so
  // ToF does NOT gate it — run on a clear floor / stand during bring-up.
  {
    const uint8_t dp = c->dpad();   // bitmask: UP=0x01 DOWN=0x02 RIGHT=0x04 LEFT=0x08
    struct DpadTurn { uint8_t bit; float heading_deg; };
    static const DpadTurn DPAD_TURNS[] = {
      {0x01,   0.0f},   // Up    -> 0
      {0x08,  90.0f},   // Left  -> +90 (CCW)
      {0x02, 180.0f},   // Down  -> 180
      {0x04, -90.0f},   // Right -> -90 (CW)
    };
    for (uint8_t i = 0; i < (uint8_t)(sizeof(DPAD_TURNS) / sizeof(DPAD_TURNS[0])); i++) {
      const uint8_t bit = DPAD_TURNS[i].bit;
      if ((dp & bit) && !(s_prev_dpad & bit)) {                  // rising edge: one turn/press
        float theta, rate;
        LOCK_STATE(); theta = g_ctx.odom.theta; rate = g_ctx.params.default_turn_rate; UNLOCK_STATE();
        const float delta_deg =
            gp_wrap_pi(gp_deg2rad(DPAD_TURNS[i].heading_deg) - theta) * 180.0f / (float)M_PI;
        ctl_manual_turn(delta_deg, rate);
      }
    }
    s_prev_dpad = dp;
  }

  // FULL-OVERRIDE: hold BOTH analog triggers near full — a deliberate two-hand gesture,
  // distinct from the L1/R1 shoulder buttons used for creep/boost.
  float br = (float)c->brake()    / GAMEPAD_TRIGGER_MAX;
  float th = (float)c->throttle() / GAMEPAD_TRIGGER_MAX;
  bool fo = (br >= GAMEPAD_FULL_OVERRIDE_FRAC && th >= GAMEPAD_FULL_OVERRIDE_FRAC);
  if (fo != s_full_override) { s_full_override = fo; ctl_set_full_override(fo); }

  // Left stick -> arcade drive; the L3-selected speed level scales the caps.
  float fwd   = -stick_norm(c->axisY());   // stick up = forward
  float turn  =  stick_norm(c->axisX());   // stick right = +x
  float scale = SPEED_LEVELS[s_speed_level];

  float max_lin, max_ang;
  LOCK_STATE(); max_lin = g_ctx.params.max_lin; max_ang = g_ctx.params.max_ang; UNLOCK_STATE();
  float lin =  fwd  * max_lin * scale;
  // Spin↔arcade BLEND, keyed off how far the stick is pushed forward/back. At (or near)
  // zero fwd: a pure spin with FULL turn authority at every speed level (breaks carpet
  // traction). As fwd grows through the GAMEPAD_SPIN_BLEND_FWD_LO..HI band, the turn
  // authority eases down to the level's scale and (via pivot_blend in hal) the inside
  // wheel's reverse allowance eases out — a spin tightens smoothly into a forward arc.
  // The old binary gate snapped authority 1.0 -> 0.15 (slow) at a 0.02 m/s threshold,
  // which felt like the turn dying the moment the stick tilted forward.
  float bt = clampf((fabsf(fwd) - GAMEPAD_SPIN_BLEND_FWD_LO) /
                    (GAMEPAD_SPIN_BLEND_FWD_HI - GAMEPAD_SPIN_BLEND_FWD_LO), 0.0f, 1.0f);
  bt = bt * bt * (3.0f - 2.0f * bt);       // smoothstep: zero slope at both edges
  const float turn_authority = GAMEPAD_SPIN_SCALE + (scale - GAMEPAD_SPIN_SCALE) * bt;
  float ang = -turn * max_ang * turn_authority;  // stick-right => -ang (REP-103: +ang = left)

  // Enter MANUAL on the first meaningful push; once manual, keep refreshing (incl. zero,
  // which feeds the drive deadman and holds the base stopped) until release/auto-return.
  // EXCEPTION: while a D-pad encoder-test turn is in flight (finite CMD_TURN), skip the
  // zero-stick deadman refresh — ctl_manual_drive would wipe the finite turn every poll.
  // A real stick push (meaningful) still takes over and cancels the turn (intended override).
  bool meaningful = (fabsf(lin) > 0.001f || fabsf(ang) > 0.001f);
  bool isManual, turnInFlight;
  LOCK_STATE();
  isManual     = (g_ctx.owner == OWNER_MANUAL);
  turnInFlight = (g_ctx.finite.kind == CMD_TURN);
  UNLOCK_STATE();
  if (meaningful || (isManual && !turnInFlight)) ctl_manual_drive(lin, ang, bt);

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
  UNLOCK_STATE();

  maybe_autoreturn();
}

#else
// ===========================================================================
// STUB — gamepad feature off. No Bluepad32 dependency; both hooks do nothing.
// ===========================================================================
void gamepad_init() {}
void gamepad_tick() {}
#endif  // MOTION_GAMEPAD_PRESENT
