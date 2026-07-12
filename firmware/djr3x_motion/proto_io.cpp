#include "proto_io.h"
#include "control.h"
#include "safety.h"
#include "config_params.h"
#include "gamepad.h"    // gamepad_notify_host_connected — hello-handshake rumble greet
#include "battery.h"    // batt_full command — host-side "charger says full" SOC sync
#include <ArduinoJson.h>
#include <math.h>

#define IS_NUM(v) ((v).is<float>() || (v).is<int>())

// ---- RX line buffer ------------------------------------------------------
static char   s_line[MOTION_MAX_LINE_BYTES];
static size_t s_len = 0;
static bool   s_overflow = false;

static void inc_errs() { LOCK_STATE(); g_ctx.errs++; UNLOCK_STATE(); }

// Quantized float for telemetry: full-precision floats serialize to 9-11 chars
// ("-6.6957545") and the 20 Hz telemetry frame was ~505 bytes — ~90% of the
// 115200-baud line's capacity, so ANY extra traffic (pad input load, acks,
// bigger in-motion numbers) backed frames up and the GUI showed stale data
// (field-logged: "telemetry froze while driving"). Emit floats at the precision
// consumers actually use. snprintf into a stack char[] — ArduinoJson COPIES
// mutable char* values into the document, so this is safe.
static void add_qf(JsonObject o, const char* key, float v, const char* fmt) {
  char b[16];
  snprintf(b, sizeof(b), fmt, (double)v);
  o[key] = serialized(b);
}

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
  if (battery_gauge_available()) caps.add("batt_full");
  doc["boot_id"] = bid;
  tx_line(doc);
}

void emit_telemetry() {
  // Snapshot under the lock, format outside it.
  MotionState st; MotionOwner ow; MotionGamepad gp; MotionFault fl;
  MotionZone z; MotionDir bd; uint32_t cs, errs; Odom od; TofMm tf; int16_t bm, bma;
  int8_t bsoc; GamepadLive gpl; WheelDiag wd; ImuState im;
  LOCK_STATE();
  st = g_ctx.state; ow = g_ctx.owner; gp = g_ctx.gamepad; fl = g_ctx.fault;
  z = g_ctx.zone; bd = g_ctx.blocked_dir; cs = g_ctx.cmd_seq; errs = g_ctx.errs;
  od = g_ctx.odom; tf = g_ctx.tof; bm = g_ctx.batt_mv; bma = g_ctx.batt_ma;
  bsoc = g_ctx.batt_soc; gpl = g_ctx.gp_live; wd = g_ctx.wheels; im = g_ctx.imu;
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
  add_qf(o, "x", od.x, "%.3f");        // mm-scale position is plenty
  add_qf(o, "y", od.y, "%.3f");
  add_qf(o, "theta", od.theta, "%.4f");  // 0.006° heading resolution
  add_qf(o, "lin", od.lin, "%.3f");
  add_qf(o, "ang", od.ang, "%.3f");
  // Per-wheel drive diagnostics: measured speed (m/s) + commanded duty, for
  // left/right asymmetry debugging (see WheelDiag in context.h).
  JsonObject w = doc["wheels"].to<JsonObject>();
  add_qf(w, "vl", wd.vl, "%.3f");
  add_qf(w, "vr", wd.vr, "%.3f");
  w["dl"] = wd.dl; w["dr"] = wd.dr;
  // ToF layout (docs §6): long front/rear pairs (fl/fr/rl/rr, ±22.5° off the axis)
  // + short left/right pairs (lf/lb/rf/rb). Keys match the GUI radar in dashboard.py.
  JsonObject t = doc["tof_mm"].to<JsonObject>();
  t["fl"] = tf.fl; t["fr"] = tf.fr; t["rl"] = tf.rl; t["rr"] = tf.rr;
  t["lf"] = tf.lf; t["lb"] = tf.lb; t["rf"] = tf.rf; t["rb"] = tf.rb;
  doc["batt_mv"] = bm;                   // -1 = no INA226 wired (host treats as unknown)
  doc["batt_ma"] = bma;                  // 0 unless a motor-ranged shunt is fitted
  doc["batt_soc"] = bsoc;                // coulomb-counted %, -1 = unknown
  doc["errs"] = errs;
  // IMU attitude (MPU-6050). Always present (stable schema): {ok:false} when no
  // sensor answered the boot probe. Angles in degrees; yaw relative to boot heading.
  JsonObject im2 = doc["imu"].to<JsonObject>();
  im2["ok"] = im.ok;
  if (im.ok) {
    add_qf(im2, "pitch", im.pitch, "%.1f");
    add_qf(im2, "roll",  im.roll,  "%.1f");
    add_qf(im2, "yaw",   im.yaw,   "%.1f");
  }
  // Live gamepad mirror for the GUI Motivator Control "physical controller" display.
  // Always present (stable schema): {connected:false} when no pad / non-gamepad build.
  JsonObject g = doc["gp"].to<JsonObject>();
  g["connected"] = gpl.connected;
  if (gpl.connected) {
    add_qf(g, "lx", gpl.lx, "%.2f");  // turn axis  -1..1 (right = +)
    add_qf(g, "ly", gpl.ly, "%.2f");  // drive axis -1..1 (stick-up = +)
    g["btn"] = gpl.btn_mask;     // pressed-button bitmask (GP_BTN_* order, gamepad.cpp)
  }
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
  e["kff"] = p.kff; e["min_duty"] = p.min_duty;
  e["breakaway_duty"] = p.breakaway_duty;
  e["accel_lin"] = p.accel_lin; e["accel_ang"] = p.accel_ang;
  e["counts_per_meter"] = p.counts_per_meter;
  e["track_width_m"] = p.track_width_m;
  e["assist_enabled"] = p.assist_enabled;
  e["assist_engage_mm"] = p.assist_engage_mm;
  e["assist_gain"] = p.assist_gain;
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
    // A host just connected (main.py startup / bench tool) — greet the operator
    // through the pad. Safe from this task: it only sets a flag; the Bluepad32
    // call happens in gamepad_tick on the loopTask. No-op in non-gamepad builds.
    gamepad_notify_host_connected();
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
  if (!strcmp(cmd, "batt_full")) {
    // Operator watched the charger taper to cutoff → sync the SOC ledger to
    // 100% (docs §5.11). Applied by the next 1 Hz battery_tick. Never gated:
    // it can't move the base, so it's accepted even under estop/manual.
    if (!battery_gauge_available()) { emit_ack(seq, false, R_UNSUPPORTED_CAP); return; }
    battery_request_mark_full();
    emit_ack(seq, true, ACK_OK);
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

  // wheel: single-wheel bring-up jog (diagnostic — NOT in the advertised caps). Raw
  // duty on ONE H-bridge for `ms` (default 1500, hard-capped), bypassing kinematics.
  // Gated like any moving command (estop/fault/manual reject). docs §5.10.
  if (!strcmp(cmd, "wheel")) {
    if (gate != ACK_OK) { emit_ack(seq, false, gate); return; }
    const char* side_s = doc["side"];
    int side;
    if      (side_s && (!strcmp(side_s, "left")  || !strcmp(side_s, "l"))) side = 0;
    else if (side_s && (!strcmp(side_s, "right") || !strcmp(side_s, "r"))) side = 1;
    else { emit_ack(seq, false, R_BAD_FIELD); return; }
    if (!IS_NUM(doc["frac"])) { emit_ack(seq, false, R_BAD_FIELD); return; }
    bool cl = false;
    float frac = clamp_flag(doc["frac"].as<float>(), -1.0f, 1.0f, cl);
    long ms_in = IS_NUM(doc["ms"]) ? doc["ms"].as<long>() : 1500;
    if (ms_in < 0) ms_in = 0;
    uint32_t ms = (uint32_t)ms_in;
    if (ms > HARDCAP_WHEEL_TEST_MS) { ms = HARDCAP_WHEEL_TEST_MS; cl = true; }
    ctl_wheel_test(side, frac, ms, seq);
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
