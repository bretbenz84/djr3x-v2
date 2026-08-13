// context.h — shared runtime state for the radar ring.
//
// One process-wide RadarContext holds everything the tasks touch, guarded by
// g_state_mux; all serial writes are serialized by g_tx_mux so NDJSON lines
// never interleave. Same concurrency contract as the drive base: keep critical
// sections short — copy a snapshot under the lock, then format/emit outside it.
// RULE: never hold g_state_mux while taking g_tx_mux (emit_* take g_tx_mux) —
// always release state, THEN emit.
#pragma once
#include <Arduino.h>
#include "protocol.h"
#include "pins.h"
#include "calib.h"
#include "ld2450.h"

// Per-sensor transport state. The pump task is the only writer; telemetry
// snapshots under the lock. frame_ms==0 means "never heard a frame" — the
// honest -1 discipline: an unwired sensor reports ok:false, never fake data.
struct RadarSensorState {
  Ld2450Frame frame;              // latest parsed frame (slots may be absent)
  uint32_t    frame_ms = 0;       // millis() at the last good frame; 0 = never
  uint32_t    frames_ok = 0;      // mirrors of the parser counters, for telemetry
  uint32_t    frames_bad = 0;
  uint32_t    bytes_dropped = 0;
  bool        cfg_ok = false;     // boot config transaction completed
  char        fw[24] = {0};       // sensor firmware version ("" = unknown)
};

struct RadarContext {
  RadarSensorState sensors[RADAR_SENSOR_COUNT];
  uint32_t errs    = 0;           // NDJSON parse/framing errors from the Mac side
  uint32_t boot_id = 0;           // random per boot (esp_random)
};

// ===== Globals (defined in djr3x_radar.ino) ================================
extern RadarContext      g_ctx;
extern SemaphoreHandle_t g_state_mux;  // recursive mutex guarding g_ctx
extern SemaphoreHandle_t g_tx_mux;     // serializes Serial writes

#define LOCK_STATE()   xSemaphoreTakeRecursive(g_state_mux, portMAX_DELAY)
#define UNLOCK_STATE() xSemaphoreGiveRecursive(g_state_mux)
