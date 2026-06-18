#include "proto_io.h"
#include "control.h"
#include "safety.h"
#include "config_params.h"
#include <ArduinoJson.h>
#include <math.h>

#define IS_NUM(v) ((v).is<float>() || (v).is<int>())

// ---- RX line buffer ------------------------------------------------------
static char   s_line[MOTION_MAX_LINE_BYTES];
static size_t s_len = 0;
static bool   s_overflow = false;

static void inc_errs() { LOCK_STATE(); g_ctx.errs++; UNLOCK_STATE(); }

// ---- TX: serialize one doc to a single atomic NDJSON line ----------------
static void tx_line(JsonDocument& doc) {
  char buf[MOTION_TX_BUF_BYTES];
  size_t n = serializeJson(doc, buf, sizeof(buf));
  if (n == 0) return;
  xSemaphoreTake(g_tx_mux, portMAX_DELAY);
  Serial.write((const uint8_t*)buf, n);
  Serial.write('\n');
  xSemaphoreGive(g_tx_mux);
}

// ===== Emitters ===========================================================
void emit_hello() {
  uint32_t bid;
  LOCK_STATE(); bid = g_ctx.boot_id; UNLOCK_STATE();
  JsonDocument doc;
  doc["v"] = MOTION_PROTO_VERSION;
  doc["type"] = "hello";
  doc["proto"] = MOTION_PROTO_VERSION;
  doc["fw"] = MOTION_FW_VERSION;
  JsonArray caps = doc["caps"].to<JsonArray>();
  caps.add("drive"); caps.add("turn"); caps.add("move"); caps.add("come"); caps.add("stop");
  doc["boot_id"] = bid;
  tx_line(doc);
}

void emit_telemetry() {
  // Snapshot under the lock, format outside it.
  MotionState st; MotionOwner ow; MotionGamepad gp; MotionFault fl;
  MotionZone z; MotionDir bd; uint32_t cs, errs; Odom od; TofMm tf; int16_t bm;
  LOCK_STATE();
  st = g_ctx.state; ow = g_ctx.owner; gp = g_ctx.gamepad; fl = g_ctx.fault;
  z = g_ctx.zone; bd = g_ctx.blocked_dir; cs = g_ctx.cmd_seq; errs = g_ctx.errs;
  od = g_ctx.odom; tf = g_ctx.tof; bm = g_ctx.batt_mv;
  UNLOCK_STATE();

  JsonDocument doc;
  doc["v"] = MOTION_PROTO_VERSION;
  doc["type"] = "telemetry";
  doc["t"] = millis();
  doc["state"] = state_str(st);
  doc["owner"] = owner_str(ow);
  doc["gamepad"] = gamepad_str(gp);
  doc["fault"] = fault_str(fl);          // nullptr -> JSON null
  doc["zone"] = zone_str(z);
  doc["blocked_dir"] = dir_str(bd);
  doc["cmd_seq"] = cs;
  JsonObject o = doc["odom"].to<JsonObject>();
  o["x"] = od.x; o["y"] = od.y; o["theta"] = od.theta; o["lin"] = od.lin; o["ang"] = od.ang;
  JsonObject t = doc["tof_mm"].to<JsonObject>();
  t["front"] = tf.front; t["rear"] = tf.rear; t["left"] = tf.left; t["right"] = tf.right;
  t["fl"] = tf.fl; t["fr"] = tf.fr; t["rl"] = tf.rl; t["rr"] = tf.rr;
  doc["batt_mv"] = bm;
  doc["errs"] = errs;
  tx_line(doc);
}

void emit_ack(uint32_t seq, bool accepted, AckReason reason) {
  JsonDocument doc;
  doc["v"] = MOTION_PROTO_VERSION;
  doc["type"] = "ack";
  doc["seq"] = seq;
  doc["accepted"] = accepted;
  doc["reason"] = ack_reason_str(reason);   // nullptr -> JSON null
  tx_line(doc);
}

void emit_config_ack(uint32_t seq, bool clamped, const MotionParams& p) {
  JsonDocument doc;
  doc["v"] = MOTION_PROTO_VERSION;
  doc["type"] = "ack";
  doc["seq"] = seq;
  doc["accepted"] = true;
  doc["reason"] = clamped ? "clamped" : (const char*)nullptr;
  JsonObject e = doc["config"].to<JsonObject>();
  e["max_lin"] = p.max_lin;
  e["max_ang"] = p.max_ang;
  e["slow_zone_m"] = p.slow_zone_m;
  e["stop_zone_m"] = p.stop_zone_m;
  e["come_stop_at_m"] = p.come_stop_at_m;
  e["default_turn_deg"] = p.default_turn_deg;
  e["default_turn_rate"] = p.default_turn_rate;
  e["watchdog_ms"] = p.watchdog_ms;
  e["drive_expiry_ms"] = p.drive_expiry_ms;
  e["manual_idle_return_secs"] = p.manual_idle_return_secs;
  e["manual_autoreturn"] = p.manual_autoreturn;
  e["kp"] = p.kp; e["ki"] = p.ki; e["kd"] = p.kd;
  e["counts_per_meter"] = p.counts_per_meter;
  e["track_width_m"] = p.track_width_m;
  tx_line(doc);
}

void emit_done(uint32_t seq, DoneResult r, const Odom& od) {
  JsonDocument doc;
  doc["v"] = MOTION_PROTO_VERSION;
  doc["type"] = "done";
  doc["seq"] = seq;
  doc["result"] = done_result_str(r);
  JsonObject o = doc["odom"].to<JsonObject>();
  o["x"] = od.x; o["y"] = od.y; o["theta"] = od.theta;
  tx_line(doc);
}

void emit_event(const char* event) {
  JsonDocument doc;
  doc["v"] = MOTION_PROTO_VERSION;
  doc["type"] = "event";
  doc["t"] = millis();
  doc["event"] = event;
  tx_line(doc);
}

void emit_event_kv(const char* event, const char* key, const char* val) {
  JsonDocument doc;
  doc["v"] = MOTION_PROTO_VERSION;
  doc["type"] = "event";
  doc["t"] = millis();
  doc["event"] = event;
  doc[key] = val;
  tx_line(doc);
}

void emit_event_boot(uint32_t boot_id) {
  JsonDocument doc;
  doc["v"] = MOTION_PROTO_VERSION;
  doc["type"] = "event";
  doc["t"] = millis();
  doc["event"] = "boot";
  doc["boot_id"] = boot_id;
  doc["fw"] = MOTION_FW_VERSION;
  tx_line(doc);
}

void emit_log(const char* lvl, const char* msg) {
  JsonDocument doc;
  doc["v"] = MOTION_PROTO_VERSION;
  doc["type"] = "log";
  doc["t"] = millis();
  doc["lvl"] = lvl;
  doc["msg"] = msg;
  tx_line(doc);
}

// ===== Command gating =====================================================
// ACK_OK if a *moving* command may run now, else the rejection reason.
static AckReason motion_gate() {
  MotionState st; MotionOwner ow;
  LOCK_STATE(); st = g_ctx.state; ow = g_ctx.owner; UNLOCK_STATE();
  if (st == ST_ESTOP) return R_ESTOP;
  if (st == ST_FAULT) return R_FAULT;
  if (ow == OWNER_MANUAL) return R_MANUAL_OVERRIDE;
  return ACK_OK;
}

static float clamp_flag(float v, float lo, float hi, bool& clamped) {
  float c = clampf(v, lo, hi);
  if (c != v) clamped = true;
  return c;
}

// ===== Dispatch ===========================================================
static void dispatch(const char* cmd, JsonDocument& doc, uint32_t seq) {
  // hello / ping are handled before gating.
  if (!strcmp(cmd, "ping")) {
    LOCK_STATE(); g_ctx.cmd_seq = seq; UNLOCK_STATE();
    return;   // never acked
  }
  if (!strcmp(cmd, "hello")) {
    emit_hello();
    return;
  }

  // Always-available regardless of motion gate:
  if (!strcmp(cmd, "stop")) {
    ctl_stop(seq); emit_ack(seq, true, ACK_OK); return;
  }
  if (!strcmp(cmd, "estop")) {
    ctl_estop(seq); emit_ack(seq, true, ACK_OK); return;
  }
  if (!strcmp(cmd, "clear")) {
    bool ok = ctl_clear(seq);
    emit_ack(seq, ok, ok ? ACK_OK : R_NOTHING_TO_CLEAR);
    return;
  }
  if (!strcmp(cmd, "config")) {
    MotionParams eff;
    bool clamped = apply_config(doc.as<JsonObjectConst>(), eff);
    emit_config_ack(seq, clamped, eff);
    return;
  }

  // Moving commands: gate first.
  AckReason gate = motion_gate();

  if (!strcmp(cmd, "drive")) {
    if (gate != ACK_OK) { emit_ack(seq, false, gate); return; }
    if (!IS_NUM(doc["lin"]) || !IS_NUM(doc["ang"])) { emit_ack(seq, false, R_BAD_FIELD); return; }
    MotionParams P; LOCK_STATE(); P = g_ctx.params; UNLOCK_STATE();
    bool cl = false;
    float lin = clamp_flag(doc["lin"].as<float>(), -P.max_lin, P.max_lin, cl);
    float ang = clamp_flag(doc["ang"].as<float>(), -P.max_ang, P.max_ang, cl);
    ctl_drive(lin, ang, seq);
    emit_ack(seq, true, cl ? R_CLAMPED : ACK_OK);
    return;
  }

  if (!strcmp(cmd, "turn")) {
    if (gate != ACK_OK) { emit_ack(seq, false, gate); return; }
    if (!IS_NUM(doc["deg"])) { emit_ack(seq, false, R_BAD_FIELD); return; }
    MotionParams P; LOCK_STATE(); P = g_ctx.params; UNLOCK_STATE();
    bool cl = false;
    float deg  = clamp_flag(doc["deg"].as<float>(), -360.0f, 360.0f, cl);
    float rate = IS_NUM(doc["rate"]) ? doc["rate"].as<float>() : P.default_turn_rate;
    rate = clamp_flag(rate, 1.0f, HARDCAP_MAX_TURN_RATE_DPS, cl);
    ctl_turn(deg, rate, seq);
    emit_ack(seq, true, cl ? R_CLAMPED : ACK_OK);
    return;
  }

  if (!strcmp(cmd, "move")) {
    if (gate != ACK_OK) { emit_ack(seq, false, gate); return; }
    if (!IS_NUM(doc["dist"])) { emit_ack(seq, false, R_BAD_FIELD); return; }
    MotionParams P; LOCK_STATE(); P = g_ctx.params; UNLOCK_STATE();
    bool cl = false;
    float dist  = clamp_flag(doc["dist"].as<float>(), -10.0f, 10.0f, cl);
    float speed = IS_NUM(doc["speed"]) ? doc["speed"].as<float>() : P.max_lin;
    speed = clamp_flag(speed, 0.0f, P.max_lin, cl);
    ctl_move(dist, speed, seq);
    emit_ack(seq, true, cl ? R_CLAMPED : ACK_OK);
    return;
  }

  if (!strcmp(cmd, "come")) {
    if (gate != ACK_OK) { emit_ack(seq, false, gate); return; }
    MotionParams P; LOCK_STATE(); P = g_ctx.params; UNLOCK_STATE();
    bool cl = false;
    float heading = IS_NUM(doc["heading"]) ? doc["heading"].as<float>() : 0.0f;
    heading = clamp_flag(heading, -180.0f, 180.0f, cl);
    float stop_at = IS_NUM(doc["stop_at"]) ? doc["stop_at"].as<float>() : P.come_stop_at_m;
    stop_at = clamp_flag(stop_at, 0.05f, 5.0f, cl);
    ctl_come(heading, stop_at, seq);
    emit_ack(seq, true, cl ? R_CLAMPED : ACK_OK);
    return;
  }

  // Unknown command.
  emit_ack(seq, false, R_UNKNOWN_CMD);
}

// ===== Line processing ====================================================
static void process_line(char* buf) {
  JsonDocument doc;
  DeserializationError err = deserializeJson(doc, buf);
  if (err) { inc_errs(); return; }

  // Envelope: require v. Missing -> drop. Wrong -> bad_version (if it's a cmd).
  if (doc["v"].isNull()) { inc_errs(); return; }
  if (doc["v"].as<int>() != MOTION_PROTO_VERSION) {
    if (doc["cmd"].is<const char*>()) {
      uint32_t seq = doc["seq"].as<uint32_t>();
      emit_ack(seq, false, R_BAD_VERSION);
    } else {
      inc_errs();
    }
    return;
  }

  const char* cmd = doc["cmd"];
  if (cmd == nullptr) { inc_errs(); return; }   // not a command we recognize

  // A valid, framed command line means the Mac is alive -> feed the watchdog
  // (and recover from comms_lost) BEFORE acting on it.
  note_mac_line();

  uint32_t seq = doc["seq"].as<uint32_t>();
  dispatch(cmd, doc, seq);
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
      if (s_len < (MOTION_MAX_LINE_BYTES - 1)) {
        s_line[s_len++] = c;
      } else {
        s_overflow = true;   // too long; will be dropped at the next '\n'
      }
    }
  }
}
