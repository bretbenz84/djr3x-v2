#include "proto_io.h"
#include <ArduinoJson.h>
#include <math.h>

// ---- RX line buffer ------------------------------------------------------
static char   s_line[RADAR_MAX_LINE_BYTES];
static size_t s_len = 0;
static bool   s_overflow = false;

static void inc_errs() { LOCK_STATE(); g_ctx.errs++; UNLOCK_STATE(); }

// Quantized float: same rationale as the drive base (full-precision floats
// bloat frames for precision no consumer uses). snprintf into a stack char[]
// — ArduinoJson COPIES mutable char* values into the document, so this is safe.
static void add_qf(JsonObject o, const char* key, float v, const char* fmt) {
  char b[16];
  snprintf(b, sizeof(b), fmt, (double)v);
  o[key] = serialized(b);
}

// ---- TX: serialize one doc to a single atomic NDJSON line ----------------
static void tx_line(JsonDocument& doc) {
  // A line that would not fit is DROPPED, never truncated — a cut JSON line
  // poisons every consumer downstream (drive-base field bug 2026-07-13).
  static char buf[RADAR_TX_BUF_BYTES];
  static bool overflow_warned = false;
  xSemaphoreTake(g_tx_mux, portMAX_DELAY);
  const bool overflow = (measureJson(doc) > sizeof(buf) - 1);
  if (!overflow) {
    size_t n = serializeJson(doc, buf, sizeof(buf));
    if (n > 0) {
      Serial.write((const uint8_t*)buf, n);
      Serial.write('\n');
    }
  }
  xSemaphoreGive(g_tx_mux);
  if (overflow && !overflow_warned) {
    overflow_warned = true;
    emit_log("warn", "proto: TX line exceeds RADAR_TX_BUF_BYTES — dropped; grow the buffer");
  }
}

// ===== Emitters ===========================================================
void emit_hello() {
  uint32_t bid;
  bool cfg[RADAR_SENSOR_COUNT];
  char fw[RADAR_SENSOR_COUNT][24];
  char bt[RADAR_SENSOR_COUNT][24];
  LOCK_STATE();
  bid = g_ctx.boot_id;
  for (int i = 0; i < RADAR_SENSOR_COUNT; i++) {
    cfg[i] = g_ctx.sensors[i].cfg_ok;
    memcpy(fw[i], g_ctx.sensors[i].fw, sizeof(fw[i]));
    memcpy(bt[i], g_ctx.sensors[i].bt, sizeof(bt[i]));
  }
  UNLOCK_STATE();
  JsonDocument doc;
  doc["v"] = RADAR_PROTO_VERSION;
  doc["type"] = "hello";
  doc["proto"] = RADAR_PROTO_VERSION;
  doc["fw"] = RADAR_FW_VERSION;
  // caps deliberately does NOT include "drive" — setup_macos.sh's motion-base
  // auto-detect probes every ESP32 with a hello and must not mistake this
  // board for the base.
  JsonArray caps = doc["caps"].to<JsonArray>();
  caps.add("radar");
  doc["boot_id"] = bid;
  // Per-sensor identity for bring-up: did the boot config transaction land,
  // what module firmware answered, and where the module's Bluetooth radio
  // stands (all empty until the real-HW build talks to a live module).
  JsonArray sens = doc["sensors"].to<JsonArray>();
  for (int i = 0; i < RADAR_SENSOR_COUNT; i++) {
    JsonObject s = sens.add<JsonObject>();
    s["mount"] = RADAR_SENSORS[i].mount_deg;
    s["cfg"] = cfg[i];
    if (fw[i][0]) s["fw"] = fw[i];
    if (bt[i][0]) s["bt"] = bt[i];
  }
  tx_line(doc);
}

void emit_telemetry() {
  // Snapshot under the lock, rotate/fuse/format outside it.
  RadarSensorState snap[RADAR_SENSOR_COUNT];
  uint32_t errs;
  LOCK_STATE();
  for (int i = 0; i < RADAR_SENSOR_COUNT; i++) snap[i] = g_ctx.sensors[i];
  errs = g_ctx.errs;
  UNLOCK_STATE();

  const uint32_t now = millis();
  RadarTargetRobot raw[RADAR_MAX_RAW_TARGETS];
  int n_raw = 0;
  int up = 0;
  bool fresh[RADAR_SENSOR_COUNT];
  for (int i = 0; i < RADAR_SENSOR_COUNT; i++) {
    fresh[i] = snap[i].frame_ms != 0 &&
               (uint32_t)(now - snap[i].frame_ms) <= RADAR_SENSOR_STALE_MS;
    if (!fresh[i]) continue;   // stale/never-heard sensors contribute nothing
    up++;
    for (int k = 0; k < LD2450_TARGET_SLOTS; k++) {
      RadarTargetRobot r;
      if (radar_local_to_robot(&snap[i].frame.t[k], i,
                               RADAR_SENSORS[i].mount_deg,
                               RADAR_FLIP_X != 0, &r)) {
        if (n_raw < RADAR_MAX_RAW_TARGETS) raw[n_raw++] = r;
      }
    }
  }
  RadarTargetRobot fused[RADAR_FUSED_MAX];
  const int n_fused = radar_fuse(raw, n_raw, fused, RADAR_FUSED_MAX);

  JsonDocument doc;
  doc["v"] = RADAR_PROTO_VERSION;
  doc["type"] = "telemetry";
  doc["t"] = now;
  // Stable schema with an ok flag (drive-base imu/env/mag pattern): the Mac
  // never key-checks. ok=false means NO sensor is delivering frames — the
  // honest "I can't see", never an empty room claim.
  JsonObject radar = doc["radar"].to<JsonObject>();
  radar["ok"] = up > 0;
  radar["up"] = up;
  JsonArray targets = radar["targets"].to<JsonArray>();
  for (int i = 0; i < n_fused; i++) {
    JsonObject t = targets.add<JsonObject>();
    add_qf(t, "b", fused[i].bearing_deg, "%.1f");   // deg, + = left/CCW
    add_qf(t, "r", fused[i].range_m,     "%.2f");   // m
    add_qf(t, "c", fused[i].confidence,  "%.2f");
    add_qf(t, "s", fused[i].speed_mps,   "%.2f");   // m/s radial
    t["m"] = fused[i].sensors;                      // contributing-sensor bitmask
  }
  // Per-sensor transport health, for bring-up and the Mac's logs.
  JsonArray sens = doc["sens"].to<JsonArray>();
  for (int i = 0; i < RADAR_SENSOR_COUNT; i++) {
    JsonObject s = sens.add<JsonObject>();
    s["ok"] = fresh[i];
    s["frames"] = snap[i].frames_ok;
    s["bad"] = snap[i].frames_bad;
    s["drop"] = snap[i].bytes_dropped;
  }
  doc["errs"] = errs;
  tx_line(doc);
}

void emit_event_boot(uint32_t boot_id) {
  JsonDocument doc;
  doc["v"] = RADAR_PROTO_VERSION;
  doc["type"] = "event";
  doc["t"] = millis();
  doc["event"] = "boot";
  doc["boot_id"] = boot_id;
  doc["fw"] = RADAR_FW_VERSION;
  tx_line(doc);
}

void emit_log(const char* lvl, const char* msg) {
  JsonDocument doc;
  doc["v"] = RADAR_PROTO_VERSION;
  doc["type"] = "log";
  doc["t"] = millis();
  doc["lvl"] = lvl;
  doc["msg"] = msg;
  tx_line(doc);
}

static void emit_ack(uint32_t seq, bool accepted, const char* reason) {
  JsonDocument doc;
  doc["v"] = RADAR_PROTO_VERSION;
  doc["type"] = "ack";
  doc["seq"] = seq;
  doc["accepted"] = accepted;
  if (reason) doc["reason"] = reason; else doc["reason"] = (const char*)nullptr;
  tx_line(doc);
}

// ===== Line processing ====================================================
static void process_line(char* buf) {
  JsonDocument doc;
  DeserializationError err = deserializeJson(doc, buf);
  if (err) { inc_errs(); return; }
  if (doc["v"].isNull()) { inc_errs(); return; }

  const char* cmd = doc["cmd"];
  if (cmd == nullptr) { inc_errs(); return; }
  uint32_t seq = doc["seq"].as<uint32_t>();

  if (doc["v"].as<int>() != RADAR_PROTO_VERSION) {
    emit_ack(seq, false, "bad_version");
    return;
  }
  if (!strcmp(cmd, "ping")) return;          // accepted, never acked (§5.1 mirror)
  if (!strcmp(cmd, "hello")) { emit_hello(); return; }
  emit_ack(seq, false, "unknown_cmd");
}

void proto_init() {
  s_len = 0;
  s_overflow = false;
}

void proto_poll() {
  while (Serial.available() > 0) {
    char c = (char)Serial.read();
    if (c == '\n') {
      if (s_overflow) {
        inc_errs();
      } else if (s_len > 0) {
        s_line[s_len] = '\0';
        process_line(s_line);
      }
      s_len = 0;
      s_overflow = false;
    } else if (c == '\r') {
      // tolerate and ignore CR
    } else {
      if (s_len < (RADAR_MAX_LINE_BYTES - 1)) {
        s_line[s_len++] = c;
      } else {
        s_overflow = true;   // too long; dropped at the next '\n'
      }
    }
  }
}
