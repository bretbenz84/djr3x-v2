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
//   L1 / R1    : creep / boost speed scale
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

static ControllerPtr s_ctl = nullptr;     // the one pad we drive from
static bool s_prev_b = false;
static bool s_prev_start = false;
static bool s_full_override = false;

static void onConnect(ControllerPtr c) {
  if (!s_ctl) { s_ctl = c; ctl_set_gamepad(true); }   // take the first pad; filter reads in tick
}

static void onDisconnect(ControllerPtr c) {
  if (s_ctl == c) {
    s_ctl = nullptr;
    s_full_override = false;
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
// ---------------------------------------------------------------------------
static uint16_t s_prev_actions = 0;

static void poll_action_buttons(ControllerPtr c) {
  const uint8_t dp = c->dpad();   // Bluepad32 dpad bitmask: UP=1 DOWN=2 RIGHT=4 LEFT=8
  struct ActionBtn { const char* name; bool pressed; };
  const ActionBtn btns[] = {
    {"a",          c->a()},
    {"x",          c->x()},
    {"y",          c->y()},
    {"dpad_up",    (bool)(dp & 0x01)},
    {"dpad_down",  (bool)(dp & 0x02)},
    {"dpad_right", (bool)(dp & 0x04)},
    {"dpad_left",  (bool)(dp & 0x08)},
    {"select",     c->miscSelect()},   // the "-" button
    {"home",       c->miscSystem()},   // the star / home button
    {"l3",         c->thumbL()},        // left stick click
    {"r3",         c->thumbR()},        // right stick click
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

  // FULL-OVERRIDE: hold BOTH analog triggers near full — a deliberate two-hand gesture,
  // distinct from the L1/R1 shoulder buttons used for creep/boost.
  float br = (float)c->brake()    / GAMEPAD_TRIGGER_MAX;
  float th = (float)c->throttle() / GAMEPAD_TRIGGER_MAX;
  bool fo = (br >= GAMEPAD_FULL_OVERRIDE_FRAC && th >= GAMEPAD_FULL_OVERRIDE_FRAC);
  if (fo != s_full_override) { s_full_override = fo; ctl_set_full_override(fo); }

  // Left stick -> arcade drive; L1 creep / R1 boost scales the caps.
  float fwd   = -stick_norm(c->axisY());   // stick up = forward
  float turn  =  stick_norm(c->axisX());   // stick right = +x
  float scale = c->l1() ? GAMEPAD_SCALE_CREEP : (c->r1() ? GAMEPAD_SCALE_BOOST : GAMEPAD_SCALE_CRUISE);

  float max_lin, max_ang;
  LOCK_STATE(); max_lin = g_ctx.params.max_lin; max_ang = g_ctx.params.max_ang; UNLOCK_STATE();
  float lin =  fwd  * max_lin * scale;
  float ang = -turn * max_ang * scale;     // stick-right => turn right => -ang (REP-103: +ang = left)

  // Enter MANUAL on the first meaningful push; once manual, keep refreshing (incl. zero,
  // which feeds the drive deadman and holds the base stopped) until release/auto-return.
  bool meaningful = (fabsf(lin) > 0.001f || fabsf(ang) > 0.001f);
  bool isManual; LOCK_STATE(); isManual = (g_ctx.owner == OWNER_MANUAL); UNLOCK_STATE();
  if (meaningful || isManual) ctl_manual_drive(lin, ang);

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
