// control.h — motion control: setpoints, the (stubbed) plant + odometry, and
// the finite-command lifecycle (turn/move/come -> `done`).
//
// All ctl_* entry points are called from the command dispatcher AFTER the
// command has been validated, clamped, and confirmed allowed (right state/owner).
// control_tick() is called from the control task at a fixed rate.
#pragma once
#include "context.h"

void control_init();
void control_tick(float dt);   // dt seconds

// Command entry points (pre-validated, pre-clamped):
void ctl_drive(float lin, float ang, uint32_t seq);          // m/s, rad/s
void ctl_turn (float deg, float rate, uint32_t seq);         // deg signed, deg/s mag
void ctl_move (float dist, float speed, uint32_t seq);       // m signed, m/s mag
void ctl_come (float heading_deg, float stop_at, uint32_t seq);
void ctl_wheel_test(int side, float frac, uint32_t ms, uint32_t seq);  // single-wheel bring-up jog (raw duty, time-bounded)
void ctl_stop (uint32_t seq);                                // controlled stop
void ctl_estop(uint32_t seq);                                // hard latch
bool ctl_clear(uint32_t seq);                                // false => nothing to clear

// Manual (gamepad) control — owner becomes MANUAL; the Mac's drive/turn/move/come are
// gated off (proto_io motion_gate) while stop/estop/config/ping still work (docs §11).
void ctl_manual_drive(float lin, float ang, float pivot_blend = 1.0f);   // m/s, rad/s, spin↔arcade blend (0 spin .. 1 arcade) — gamepad teleop setpoint
void ctl_manual_turn (float deg, float rate);   // deg signed, deg/s mag — gamepad spin-in-place BY deg as a MANUAL finite turn (D-pad Left/Right nudge + L1 encoder test)
void ctl_manual_move (float dist, float speed); // m signed, m/s mag — gamepad forward/back BY dist as a MANUAL finite move (D-pad Up/Down nudge; ToF-gated like any move)
void ctl_manual_stop();                         // stop but STAY manual (disconnect failsafe)
void ctl_manual_release();                      // stop and hand control back to AUTO
void ctl_set_full_override(bool on);            // gamepad bypasses ToF gating while held
void ctl_set_gamepad(bool connected);           // telemetry: paired-pad link status
