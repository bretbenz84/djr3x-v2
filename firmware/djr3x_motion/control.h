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
void ctl_stop (uint32_t seq);                                // controlled stop
void ctl_estop(uint32_t seq);                                // hard latch
bool ctl_clear(uint32_t seq);                                // false => nothing to clear
