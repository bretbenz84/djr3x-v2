// proto_io.h — NDJSON framing to the Mac, command parse, and all wire emitters.
//
// proto_poll() drains the USB-CDC RX, frames NDJSON lines (<=512 B, drop-on-
// overflow), and answers the tiny command set a sensor board needs: `hello`
// (identify — the Mac AND setup_macos.sh's board-discrimination probe read the
// caps) and `ping` (accepted, ignored — there is no watchdog to feed on a
// board that can't move anything). Everything else acks unknown_cmd.
//
// The emit_* helpers serialize one JSON line atomically under g_tx_mux — call
// them only while NOT holding g_state_mux (context.h lock-order rule).
#pragma once
#include "context.h"
#include "fusion.h"

void proto_init();
void proto_poll();        // call frequently from the serial task

// Emitters (each takes g_tx_mux internally):
void emit_hello();
void emit_telemetry();    // snapshots sensors, fuses, emits — the 10 Hz heartbeat
void emit_event_boot(uint32_t boot_id);
void emit_log(const char* lvl, const char* msg);   // lvl: debug|info|warn|error
