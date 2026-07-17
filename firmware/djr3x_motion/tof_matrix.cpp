// tof_matrix.cpp — DFRobot 8x8 Matrix ToF (SEN0628) front obstacle sensor.
// See tof_matrix.h for the role; calib.h ("8x8 Matrix ToF") for every knob.
//
// ---- Why a vendored mini-driver instead of DFRobot_MatrixLidar ---------------
// The official library is fine on the bench (firmware/tof_matrix_test uses it)
// but is not safe on the robot's 50 Hz sensor task:
//   - its receive path busy-polls the bus with a PRIVATE, unsettable 8000 ms
//     timeout — a wedged RP2040 would stall the sensor task for 8 s (the exact
//     I2C-stall → starved-Bluetooth failure class already fought once on this
//     bus, see djr3x_motion.ino task notes);
//   - it mallocs/frees every call (heap churn in a hot loop);
//   - setRangingMode() hides a blocking delay(5000) we need under FreeRTOS
//     control (vTaskDelay on a background task, not a boot stall).
// The wire protocol is tiny and fully visible in the library source, so we speak
// it directly with hard deadlines and static buffers. Bench firmware keeps using
// the library; only the robot build uses this driver.
//
// ---- Wire protocol (confirmed against DFRobot_MatrixLidar 1.x source) --------
//   Mac->sensor I2C write:  {0x55, argsNumH, argsNumL, cmd, args...}
//                           argsNum = len(args) + 1 (the +1 is the lib's quirk —
//                           copied verbatim; the RP2040 expects it).
//     CMD_SETMODE (1): args {0,0,0, matrix} with matrix=8 for 8x8. Needs ~5 s
//                      settle after the ACK before frames are valid.
//     CMD_ALLDATA (2): no args. Response payload = 128 bytes: 64x uint16 LE
//                      millimetres, row-major (index = y*8 + x). 0 = no return.
//   Response (I2C reads): poll one status byte until != 0xFF —
//     0x53 = success, 0x63 = failed. Then {cmd, lenL, lenH, payload[len]},
//     reads chunked <= 32 bytes (the RP2040's I2C cache size).
#include "hal.h"

#if MOTION_TOF_MATRIX_PRESENT
#include "pins.h"
#include "calib.h"
#include "proto_io.h"
#include <Arduino.h>
#include <Wire.h>
#include <math.h>

// ---- Protocol constants (mirrors the DFRobot library source) ----------------
static const uint8_t MX_HEAD          = 0x55;
static const uint8_t MX_CMD_SETMODE   = 1;
static const uint8_t MX_CMD_ALLDATA   = 2;
static const uint8_t MX_STATUS_OK     = 0x53;
static const uint8_t MX_STATUS_FAILED = 0x63;
static const uint8_t MX_STATUS_IDLE   = 0xFF;   // "response not ready yet"
static const int     MX_I2C_CHUNK     = 32;     // RP2040-side I2C cache limit
static const uint8_t MX_MODE_8X8      = 8;      // eMatrix_8X8

// ---- State (init task writes once; sensor task owns everything after) -------
static volatile bool s_ready = false;           // init task -> sensor task handoff

static uint16_t s_grid[64];                     // latest raw frame, row-major mm
static volatile int16_t  s_fl = -1, s_fr = -1;  // published aggregates (mm; -1 = no data)
static volatile uint32_t s_pub_ms = 0;          // when s_fl/s_fr were last refreshed
static float    s_filt_fl = -1.0f, s_filt_fr = -1.0f;  // fast-attack/slow-release state
static uint8_t  s_err_streak = 0;               // consecutive failed frame reads

// Per-row floor-rejection tables, precomputed on first use (row 0 = physically TOP
// after TOF_MATRIX_FLIP_V normalization). reject_mm = readings at/beyond this are
// floor (or too far to matter) -> clear; cos_elev projects a ray reading onto the
// horizontal plane the zone logic reasons in.
static bool  s_geom_ready = false;
static float s_row_reject_mm[8];
static float s_row_cos[8];

static void mx_compute_geometry() {
  const float h_mm   = TOF_MATRIX_HEIGHT_M * 1000.0f;
  const float pitch  = TOF_MATRIX_PITCH_DEG;
  const float rowdeg = TOF_MATRIX_VFOV_DEG / 8.0f;
  for (int r = 0; r < 8; r++) {
    // Row-centre elevation above the horizon: top row (r=0) looks UP, bottom
    // row (r=7) looks DOWN, level mount => symmetric about 0.
    const float elev_deg = (3.5f - (float)r) * rowdeg + pitch;
    const float elev_rad = elev_deg * (float)M_PI / 180.0f;
    s_row_cos[r] = cosf(elev_rad);
    if (elev_deg < -0.5f) {
      // Looking down: the empty floor returns h/sin(|elev|) along the ray. A
      // reading at/beyond FLOOR_TOLERANCE of that is the floor itself (or
      // near-floor clutter the base can shrug off); meaningfully SHORTER is a
      // real obstacle standing proud of the floor. FLOOR_MIN_MM is the
      // belt-and-braces lower bound: nothing that close is ever "floor", even
      // if the configured mount height is a little off.
      const float floor_ray_mm = h_mm / sinf(-elev_rad);
      float reject = floor_ray_mm * TOF_MATRIX_FLOOR_TOLERANCE;
      if (reject < (float)TOF_MATRIX_FLOOR_MIN_MM) reject = (float)TOF_MATRIX_FLOOR_MIN_MM;
      s_row_reject_mm[r] = reject;
    } else {
      // At/above the horizon: no floor in view — every valid return is a
      // candidate obstacle (walls, chair seats/backs, people).
      s_row_reject_mm[r] = 1.0e9f;
    }
  }
  s_geom_ready = true;
}

static void mx_aggregate(int16_t* out_fl, int16_t* out_fr);
static int16_t mx_filter(float* state, int16_t mm);

// ---- Bounded I2C primitives (hardware I2C1 via Wire1) -------------------------
// The matrix lives ALONE on the second I2C controller (GPIO4/5, pins.h) — never
// on the 21/22 trunk with the INA226/IMU. Every transaction runs on the matrix's
// own poll task, so a slow or stretching sensor costs only this task's time.
static bool mx_probe() {
  Wire1.beginTransmission(TOF_MATRIX_ADDR);
  return Wire1.endTransmission() == 0;
}

static bool mx_send_cmd(uint8_t cmd, const uint8_t* args, uint8_t args_len) {
  Wire1.beginTransmission(TOF_MATRIX_ADDR);
  Wire1.write(MX_HEAD);
  Wire1.write((uint8_t)(((args_len + 1) >> 8) & 0xFF));  // argsNumH (lib's len+1 quirk)
  Wire1.write((uint8_t)((args_len + 1) & 0xFF));         // argsNumL
  Wire1.write(cmd);
  if (args_len) Wire1.write(args, args_len);
  const bool ok = (Wire1.endTransmission() == 0);
  return ok;
}

// Read len bytes in <=32-byte chunks. Returns false on a short/NACKed read.
static bool mx_read_bytes(uint8_t* dst, int len) {
  while (len > 0) {
    const int n = (len > MX_I2C_CHUNK) ? MX_I2C_CHUNK : len;
    if ((int)Wire1.requestFrom((uint8_t)TOF_MATRIX_ADDR, (uint8_t)n) != n) return false;
    for (int i = 0; i < n; i++) {
      const int b = Wire1.read();
      if (b < 0) return false;
      *dst++ = (uint8_t)b;
    }
    len -= n;
  }
  return true;
}

// Await + read one response packet for `cmd` within deadline_ms. On success the
// payload (up to max_payload bytes) is in payload_out and its length returned;
// -1 = timeout / protocol error / failed status. HARD-BOUNDED: this is what
// replaces the library's private 8 s busy-poll.
static int mx_recv_response(uint8_t cmd, uint8_t* payload_out, int max_payload,
                            uint32_t deadline_ms) {
  const uint32_t t0 = millis();
  for (;;) {
    uint8_t status;
    if (!mx_read_bytes(&status, 1)) status = MX_STATUS_IDLE;   // NACK = not ready
    if (status == MX_STATUS_OK || status == MX_STATUS_FAILED) {
      uint8_t hdr[3];                                          // {cmd, lenL, lenH}
      if (!mx_read_bytes(hdr, 3)) return -1;
      if (hdr[0] != cmd) return -1;                            // response for someone else
      int len = (int)hdr[1] | ((int)hdr[2] << 8);
      if (len < 0 || len > 1000) return -1;                    // insane length — bail
      // Always DRAIN the payload (even on FAILED / oversize) so the RP2040's
      // response buffer isn't left half-read for the next transaction.
      uint8_t scratch[MX_I2C_CHUNK];
      int want = (len <= max_payload) ? len : max_payload;
      if (want > 0 && !mx_read_bytes(payload_out, want)) return -1;
      for (int left = len - want; left > 0; ) {
        const int n = (left > MX_I2C_CHUNK) ? MX_I2C_CHUNK : left;
        if (!mx_read_bytes(scratch, n)) return -1;
        left -= n;
      }
      return (status == MX_STATUS_OK) ? want : -1;
    }
    if ((uint32_t)(millis() - t0) >= deadline_ms) return -1;
    vTaskDelay(pdMS_TO_TICKS(2));    // yield the core between status polls
  }
}

// ---- Deferred init task -------------------------------------------------------
// begin-probe (retry with backoff) -> SETMODE 8x8 -> ~5 s settle -> ready.
// Runs at priority 1 (below the sensor task) so a missing/flaky sensor can retry
// forever without costing anyone anything. The Arduino-ESP32 TwoWire lock makes
// per-transaction bus sharing with the sensor task (radial ToF / IMU / INA226)
// safe; interleaved transactions to a different address are protocol-fine.
static void mx_init_task(void*) {
  uint32_t attempt = 0;
  for (;;) {
    if (mx_probe()) break;
    if ((attempt++ % 10) == 0) {
      char buf[80];
      snprintf(buf, sizeof(buf),
               "tof_matrix: no ACK at 0x%02X (attempt %lu) - check wiring/DIP addr",
               TOF_MATRIX_ADDR, (unsigned long)(attempt));
      emit_log("warn", buf);
    }
    vTaskDelay(pdMS_TO_TICKS(1000));
  }

  // ⚠ DO NOT status-poll the sensor while it reconfigures. The RP2040
  // clock-stretches during the ~5 s VL53L7CX mode switch, and the ESP32's I2C
  // hardware tolerates at most ~13 ms of stretch — hammering status reads into
  // that window wedges the shared I2C trunk and hangs the whole firmware
  // (field-observed 2026-07-16: telemetry died seconds after boot, 4/5 resets).
  // Instead: send SETMODE, wait out the reconfigure BLINDLY, then read the ack
  // exactly when the sensor is guaranteed idle again.
  for (;;) {
    const uint8_t args[4] = {0, 0, 0, MX_MODE_8X8};
    mx_send_cmd(MX_CMD_SETMODE, args, sizeof(args));
    vTaskDelay(pdMS_TO_TICKS(TOF_MATRIX_MODE_SETTLE_MS));
    uint8_t scratch[8];
    if (mx_recv_response(MX_CMD_SETMODE, scratch, sizeof(scratch),
                         TOF_MATRIX_MODE_ACK_TIMEOUT_MS) >= 0) {
      break;
    }
    emit_log("warn", "tof_matrix: 8x8 mode-set not acknowledged - retrying");
    vTaskDelay(pdMS_TO_TICKS(1000));
  }

  mx_compute_geometry();
  s_ready = true;
  emit_log("info", "tof_matrix: 8x8 front matrix ready (floor rejection active)");

  // ---- Poll loop: this task OWNS every matrix I2C transaction from here on. ----
  // WEDGE ISOLATION (field bug 2026-07-16): the RP2040 clock-stretches beyond
  // what the ESP32 I2C driver tolerates, and a recovery-path deadlock inside
  // Wire can block its caller FOREVER. When the frame reads ran inline on the
  // 50 Hz sensor task, one wedged transaction froze the sensor task -> the
  // state-lock chain -> control/telemetry/serial: the whole firmware went
  // silent (both cores' output dead). Confining the I2C to this dedicated task
  // means a wedge kills ONLY the matrix; tof_matrix_read() then times out via
  // s_pub_ms staleness and publishes an honest -1 while the robot stays alive.
  for (;;) {
    mx_send_cmd(MX_CMD_ALLDATA, nullptr, 0);
    uint8_t payload[128];
    const int got = mx_recv_response(MX_CMD_ALLDATA, payload, sizeof(payload),
                                     TOF_MATRIX_READ_TIMEOUT_MS);
    if (got == (int)sizeof(payload)) {
      s_err_streak = 0;
      for (int i = 0; i < 64; i++) {                     // uint16 little-endian, row-major
        s_grid[i] = (uint16_t)payload[2 * i] | ((uint16_t)payload[2 * i + 1] << 8);
      }
      int16_t raw_fl, raw_fr;
      mx_aggregate(&raw_fl, &raw_fr);
      s_fl = mx_filter(&s_filt_fl, raw_fl);
      s_fr = mx_filter(&s_filt_fr, raw_fr);
      s_pub_ms = millis();
    } else {
      // Failed/short frame: hold the last-good aggregates through a transient
      // error, then publish an honest -1 (same policy + streak as tof.cpp, same
      // documented fail-open in safety.cpp).
      if (s_err_streak < 255) s_err_streak++;
      if (s_err_streak == TOF_ERR_STREAK_STALE && (s_fl != -1 || s_fr != -1)) {
        s_fl = s_fr = -1;
        s_filt_fl = s_filt_fr = -1.0f;
        emit_log("warn", "tof_matrix: consecutive read errors - reporting -1");
      } else if (s_err_streak > TOF_ERR_STREAK_STALE) {
        s_fl = s_fr = -1;
      }
      s_pub_ms = millis();   // an honest error is still a live publisher
    }
    vTaskDelay(pdMS_TO_TICKS(TOF_MATRIX_FRAME_INTERVAL_MS));
  }
}

// ---- Aggregation: 64 zones -> nearest obstacle per half ----------------------
// Pure function over the raw grid + precomputed geometry. Orientation is
// normalized here: after TOF_MATRIX_FLIP_V/H, row 0 = physically TOP and col 0 =
// the ROBOT'S LEFT edge of the FOV (verify both on the bench — see calib.h).
static void mx_aggregate(int16_t* out_fl, int16_t* out_fr) {
  float best_left = 1.0e9f, best_right = 1.0e9f;
  for (int r = 0; r < 8; r++) {
    const int rr = TOF_MATRIX_FLIP_V ? (7 - r) : r;
    for (int c = 0; c < 8; c++) {
      const int cc = TOF_MATRIX_FLIP_H ? (7 - c) : c;
      const uint16_t v = s_grid[rr * 8 + cc];
      if (v == 0) continue;                              // no valid return in this zone
      if (v >= TOF_MATRIX_CLEAR_MM) continue;            // sensor's "no return" marker
                                                         // (reports 4000 for nothing seen)
      if (v < TOF_MATRIX_MIN_MM) continue;               // sub-min-range speckle
      if ((float)v >= s_row_reject_mm[r]) continue;      // the floor (or beyond it)
      const float horiz = (float)v * s_row_cos[r];       // project ray -> horizontal
      if (c < 4) { if (horiz < best_left)  best_left  = horiz; }
      else       { if (horiz < best_right) best_right = horiz; }
    }
  }
  *out_fl = (best_left  < 1.0e9f) ? (int16_t)(best_left  + 0.5f) : (int16_t)TOF_MATRIX_CLEAR_MM;
  *out_fr = (best_right < 1.0e9f) ? (int16_t)(best_right + 0.5f) : (int16_t)TOF_MATRIX_CLEAR_MM;
}

// Fast-attack / slow-release (same shape as tof.cpp's per-sensor filter): believe
// a NEARER reading instantly — never filter danger — release toward a farther one
// at TOF_RELEASE_STEP_MM per frame, so an edge-of-beam chair leg holds steady
// instead of strobing the zone.
static int16_t mx_filter(float* state, int16_t mm) {
  if (*state < 0.0f || (float)mm <= *state) {
    *state = (float)mm;
  } else {
    const float rise = (float)mm - *state;
    *state += (rise < (float)TOF_RELEASE_STEP_MM) ? rise : (float)TOF_RELEASE_STEP_MM;
  }
  return (int16_t)(*state + 0.5f);
}

// ---- Public API ---------------------------------------------------------------
void tof_matrix_init() {
  // Second controller, dedicated pins (pins.h): the matrix shares a bus with
  // NOTHING, and all its transactions run on its own expendable poll task.
  Wire1.begin(PIN_MX_I2C_SDA, PIN_MX_I2C_SCL);
  Wire1.setTimeOut(500);   // ride out slave clock-stretch; only this task waits
  emit_log("info", "tof_matrix: init deferred (mode switch needs ~5s settle)");
  xTaskCreatePinnedToCore(mx_init_task, "tofmx", 3072, nullptr, 1, nullptr, 1);
}

bool tof_matrix_ready() { return s_ready; }

// Lock-free snapshot for the 50 Hz sensor task: NO I2C here (see the wedge-
// isolation note in the poll loop). If the poll task dies mid-transaction the
// publishes stop; the staleness bound below converts that silence into an
// honest -1 (and one log line) instead of freezing distances at their last
// value while the base keeps driving.
void tof_matrix_read(int16_t* fl, int16_t* fr) {
  if (!s_ready) { *fl = -1; *fr = -1; return; }

  if ((uint32_t)(millis() - s_pub_ms) > TOF_MATRIX_STALE_MS) {
    static bool s_warned = false;
    if (!s_warned) {
      s_warned = true;
      emit_log("warn", "tof_matrix: publisher stale (poll task wedged?) - reporting -1");
    }
    *fl = -1; *fr = -1;
    return;
  }

  *fl = s_fl;
  *fr = s_fr;
}

#else
// Matrix not present in this build: header-declared symbols still exist so tof.cpp
// can reference them unconditionally if ever needed (it doesn't today — its calls
// are compiled out), and a stray call is harmless.
void tof_matrix_init() {}
bool tof_matrix_ready() { return false; }
void tof_matrix_read(int16_t* fl, int16_t* fr) { *fl = -1; *fr = -1; }
#endif  // MOTION_TOF_MATRIX_PRESENT
