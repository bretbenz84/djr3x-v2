// protocol.h — DJ-R3X radar-ring wire protocol v1.
//
// Single source of truth for the on-the-wire vocabulary of the radar board
// (bearing-prior ring, docs/radar-bearing-prior-spec.md). The link is the same
// NDJSON contract as the drive base (docs/motion_protocol.md §1-§2): one flat
// JSON object per line, `v` on every message, Mac→board messages carry `cmd`,
// board→Mac carry `type`, unknown fields ignored. This board is a SENSOR — it
// never moves anything — so there is no watchdog, no ack/seq machinery beyond
// politeness, and telemetry streams from boot without waiting for a handshake
// (bring-up friendliness: `arduino-cli monitor` shows targets immediately).
//
// Pure declarations; no state. Safe to include anywhere.
#pragma once
#include <stdint.h>

// ---- Version ---------------------------------------------------------------
#define RADAR_PROTO_VERSION 1
#define RADAR_FW_VERSION    "0.1.0"

// ---- Wire limits -----------------------------------------------------------
// Same drop-don't-truncate policy as the drive base (protocol.h there records
// the 2026-07-13 truncation field bug). The radar frame is small (~350 B with
// all 9 raw slots full), but native USB CDC costs nothing to keep the margin.
#define RADAR_MAX_LINE_BYTES 512   // a longer RX line is dropped through the next '\n'
#define RADAR_TX_BUF_BYTES   1024  // serialized output line buffer; too-long = dropped
