// proto_io.h — serial framing, command parse+dispatch, and all wire emitters.
//
// proto_poll() drains the serial RX, frames NDJSON lines (<=512 B, drop-on-
// overflow), parses, and dispatches commands into control/config. The emit_*
// helpers serialize one JSON line atomically under g_tx_mux — call them only
// while NOT holding g_state_mux (see context.h lock-order rule).
#pragma once
#include "context.h"
#include "protocol.h"

void proto_init();
void proto_poll();        // call frequently from the serial task

// Emitters (each takes g_tx_mux internally):
void emit_hello();
void emit_telemetry();
void emit_ack(uint32_t seq, bool accepted, AckReason reason);
void emit_config_ack(uint32_t seq, bool clamped, const MotionParams& p);
void emit_done(uint32_t seq, DoneResult r, const Odom& od);
void emit_event(const char* event);                                 // no payload
void emit_event_kv(const char* event, const char* key, const char* val);
void emit_event_boot(uint32_t boot_id);
void emit_log(const char* lvl, const char* msg);
