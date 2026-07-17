// protocol.h — DJ-R3X motion wire protocol v1 (see docs/motion_protocol.md)
//
// Single source of truth for the on-the-wire vocabulary: version, capability
// advertisement, and the enum<->string maps for every field. If this disagrees
// with docs/motion_protocol.md, the doc wins — update this to match.
//
// Pure declarations + inline helpers; no state. Safe to include anywhere.
#pragma once
#include <Arduino.h>

// ---- Version ---------------------------------------------------------------
#define MOTION_PROTO_VERSION 1
#define MOTION_FW_VERSION    "0.1.0-skeleton"

// Capabilities advertised in the `hello` reply. The skeleton runs the full
// command set against a stubbed plant, so it advertises everything. As real
// hardware lands, this list stays the same — the commands already work.
#define MOTION_CAPS_JSON     "[\"drive\",\"turn\",\"move\",\"come\",\"stop\"]"

// ---- Wire limits -----------------------------------------------------------
#define MOTION_MAX_LINE_BYTES 512   // a longer line is dropped through the next '\n'
#define MOTION_TX_BUF_BYTES   1024  // serialized output line buffer. Must hold the
                                    // LARGEST telemetry frame: with a gamepad
                                    // connected the gp block pushed frames past the
                                    // old 512 and serializeJson TRUNCATED every
                                    // line mid-JSON — consumers (menu bar, GUI)
                                    // rejected 100% of frames and froze (field bug
                                    // 2026-07-13, "stale while pad connected").

// ---- Enumerations (mirror docs/motion_protocol.md §9) ----------------------
enum MotionState : uint8_t {
  ST_IDLE = 0, ST_MOVING, ST_BLOCKED, ST_ESTOP, ST_FAULT, ST_COMMS_LOST
};
enum MotionOwner : uint8_t { OWNER_AUTO = 0, OWNER_MANUAL };
enum MotionGamepad : uint8_t { GP_NONE = 0, GP_CONNECTED };
enum MotionFault : uint8_t {
  F_NONE = 0, F_ENCODER_STALL, F_OVERCURRENT, F_TOF_ERROR, F_LOW_BATT, F_COMMS_LOST
};
enum MotionZone : uint8_t { Z_CLEAR = 0, Z_SLOW, Z_STOP, Z_CLIFF };
enum MotionDir : uint8_t { DIR_NONE = 0, DIR_FRONT, DIR_REAR, DIR_LEFT, DIR_RIGHT, DIR_BOTH };

// done.result
enum DoneResult : uint8_t {
  DONE_COMPLETED = 0, DONE_BLOCKED, DONE_ABORTED, DONE_SUPERSEDED, DONE_ESTOPPED
};

// ack.reason (ACK_OK => emitted as JSON null)
enum AckReason : uint8_t {
  ACK_OK = 0, R_CLAMPED, R_MANUAL_OVERRIDE, R_ESTOP, R_FAULT, R_UNKNOWN_CMD,
  R_BAD_FIELD, R_BAD_VERSION, R_NOTHING_TO_CLEAR, R_UNSUPPORTED_CAP
};

// Finite command kind (control-layer bookkeeping; not a wire enum).
// CMD_WHEEL is the single-wheel bring-up jog (raw duty on ONE H-bridge, time-bounded;
// docs/motion_protocol.md §5.10) — a diagnostic, NOT part of the advertised caps.
enum CmdKind : uint8_t { CMD_NONE = 0, CMD_DRIVE, CMD_TURN, CMD_MOVE, CMD_COME, CMD_WHEEL };

// ---- enum -> wire string ---------------------------------------------------
inline const char* state_str(MotionState s) {
  switch (s) {
    case ST_IDLE: return "idle";
    case ST_MOVING: return "moving";
    case ST_BLOCKED: return "blocked";
    case ST_ESTOP: return "estop";
    case ST_FAULT: return "fault";
    case ST_COMMS_LOST: return "comms_lost";
  }
  return "idle";
}
inline const char* owner_str(MotionOwner o) { return o == OWNER_MANUAL ? "manual" : "auto"; }
inline const char* gamepad_str(MotionGamepad g) { return g == GP_CONNECTED ? "connected" : "none"; }
inline const char* zone_str(MotionZone z) {
  switch (z) {
    case Z_CLEAR: return "clear";
    case Z_SLOW: return "slow";
    case Z_STOP: return "stop";
    case Z_CLIFF: return "cliff";
  }
  return "clear";
}
inline const char* dir_str(MotionDir d) {
  switch (d) {
    case DIR_NONE: return "none";
    case DIR_FRONT: return "front";
    case DIR_REAR: return "rear";
    case DIR_LEFT: return "left";
    case DIR_RIGHT: return "right";
    case DIR_BOTH: return "both";
  }
  return "none";
}
// fault: F_NONE -> nullptr (caller emits JSON null)
inline const char* fault_str(MotionFault f) {
  switch (f) {
    case F_NONE: return nullptr;
    case F_ENCODER_STALL: return "encoder_stall";
    case F_OVERCURRENT: return "overcurrent";
    case F_TOF_ERROR: return "tof_error";
    case F_LOW_BATT: return "low_batt";
    case F_COMMS_LOST: return "comms_lost";
  }
  return nullptr;
}
inline const char* done_result_str(DoneResult r) {
  switch (r) {
    case DONE_COMPLETED: return "completed";
    case DONE_BLOCKED: return "blocked";
    case DONE_ABORTED: return "aborted";
    case DONE_SUPERSEDED: return "superseded";
    case DONE_ESTOPPED: return "estopped";
  }
  return "completed";
}
// ack.reason: ACK_OK -> nullptr (caller emits JSON null)
inline const char* ack_reason_str(AckReason r) {
  switch (r) {
    case ACK_OK: return nullptr;
    case R_CLAMPED: return "clamped";
    case R_MANUAL_OVERRIDE: return "manual_override";
    case R_ESTOP: return "estop";
    case R_FAULT: return "fault";
    case R_UNKNOWN_CMD: return "unknown_cmd";
    case R_BAD_FIELD: return "bad_field";
    case R_BAD_VERSION: return "bad_version";
    case R_NOTHING_TO_CLEAR: return "nothing_to_clear";
    case R_UNSUPPORTED_CAP: return "unsupported_cap";
  }
  return nullptr;
}
