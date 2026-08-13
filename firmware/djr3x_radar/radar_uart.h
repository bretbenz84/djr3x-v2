// radar_uart.h — owner of the three LD2450 sensor UARTs (or their stub).
//
// One module owns the bus lifecycle (the i2c_trunk pattern from the drive
// base): nothing else touches a HardwareSerial or re-begins a port. The pump
// feeds each sensor's bytes through its own Ld2450Parser and publishes parsed
// frames + counters into g_ctx under the state lock; consumers (emit_telemetry)
// only ever read snapshots.
//
// RADAR_HW_PRESENT gates the real UART drivers. While it is 0 (no modules
// wired — they arrive tomorrow) the stub SYNTHESIZES a scripted scene, encodes
// it into real LD2450 wire bytes, and feeds those through the SAME parsers —
// so the whole parse → rotate → fuse → emit pipeline runs on a bare S3 and the
// Mac side is developable against a live board today. The stub is labelled in
// its boot log line; it never pretends to be real coverage.
#pragma once
#include "context.h"

#ifndef RADAR_HW_PRESENT
#define RADAR_HW_PRESENT 0    // override per-build: -DRADAR_HW_PRESENT=1
#endif

void radar_uart_init();   // begin ports + boot config transaction (real) / seed the synth scene (stub)
void radar_uart_pump();   // drain RX through the parsers, publish frames (call from the sensor task)
