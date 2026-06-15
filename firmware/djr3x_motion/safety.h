// safety.h — the reflexes: heartbeat watchdog, ToF zone evaluation + reflex
// stop, and comms-loss handling. Runs every control tick, independent of the
// Mac. note_mac_line() is called on every valid inbound line to feed the
// watchdog and recover from comms_lost.
#pragma once
#include "context.h"

void safety_init();
void safety_tick();
void note_mac_line();
