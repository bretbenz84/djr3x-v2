#include "radar_uart.h"
#include "proto_io.h"
#include "fusion.h"
#include <math.h>

static Ld2450Parser s_parser[RADAR_SENSOR_COUNT];

// Publish one parsed frame + the parser's counters (shared by real + stub).
static void publish_frame(int i, const Ld2450Frame& f) {
  LOCK_STATE();
  RadarSensorState& s = g_ctx.sensors[i];
  s.frame         = f;
  s.frame_ms      = millis();
  s.frames_ok     = s_parser[i].frames_ok;
  s.frames_bad    = s_parser[i].frames_bad;
  s.bytes_dropped = s_parser[i].bytes_dropped;
  UNLOCK_STATE();
}

#if RADAR_HW_PRESENT
// ===========================================================================
// REAL HARDWARE — three LD2450 modules on hardware UARTs (pins.h table).
// ===========================================================================
#include <HardwareSerial.h>

static HardwareSerial* s_port[RADAR_SENSOR_COUNT];

// One config-mode transaction with deadline-bounded ACK reads. A sensor that
// never ACKs the enable is skipped (absent / still booting) — boot must not
// stall the protocol for a missing module.
static bool cfg_transact(int i, uint16_t word, const uint8_t* value, size_t vlen,
                         uint8_t* ack_val, size_t* ack_len) {
  uint8_t cmd[24];
  const size_t n = ld2450_build_cmd(word, value, vlen, cmd);
  s_port[i]->write(cmd, n);
  s_port[i]->flush();

  uint8_t buf[192];
  size_t  got = 0;
  const uint32_t deadline = millis() + RADAR_CFG_ACK_TIMEOUT_MS;
  while ((int32_t)(deadline - millis()) > 0) {
    while (s_port[i]->available() > 0 && got < sizeof(buf)) {
      buf[got++] = (uint8_t)s_port[i]->read();
    }
    const uint8_t* val; size_t vl;
    if (ld2450_find_ack(buf, got, word, &val, &vl)) {
      if (vl < 2 || (val[0] | (val[1] << 8)) != 0) return false;   // status != ok
      if (ack_val && ack_len) {
        const size_t cp = vl - 2 < *ack_len ? vl - 2 : *ack_len;
        memcpy(ack_val, val + 2, cp);
        *ack_len = cp;
      }
      return true;
    }
    delay(2);   // init runs before the tasks exist; plain delay is fine here
  }
  return false;
}

// Boot config per sensor (calib.h RADAR_SENSOR_BOOT_CONFIG): read the module
// firmware version into the logs and force multi-target tracking. Read-only
// otherwise — nothing persistent is changed.
static void boot_config(int i) {
  char msg[96];
  if (!cfg_transact(i, LD2450_CMD_ENABLE_CONFIG, (const uint8_t*)"\x01\x00", 2,
                    nullptr, nullptr)) {
    snprintf(msg, sizeof(msg), "radar[%d] uart%u: no config ACK — module absent or still booting",
             i, RADAR_SENSORS[i].uart);
    emit_log("warn", msg);
    return;
  }
  uint8_t fwv[8]; size_t fwlen = sizeof(fwv);
  char fw[24] = {0};
  if (cfg_transact(i, LD2450_CMD_READ_FW, nullptr, 0, fwv, &fwlen) && fwlen >= 8) {
    // value: fw type(2) + major(2 LE) + minor(4 LE), e.g. V1.02.22062416.
    const uint16_t major = (uint16_t)(fwv[2] | (fwv[3] << 8));
    const uint32_t minor = (uint32_t)fwv[4] | ((uint32_t)fwv[5] << 8) |
                           ((uint32_t)fwv[6] << 16) | ((uint32_t)fwv[7] << 24);
    snprintf(fw, sizeof(fw), "V%u.%02u.%08lx", major >> 8, major & 0xFF,
             (unsigned long)minor);
  }
  const bool multi_ok = cfg_transact(i, LD2450_CMD_MULTI_TARGET, nullptr, 0,
                                     nullptr, nullptr);
  cfg_transact(i, LD2450_CMD_END_CONFIG, nullptr, 0, nullptr, nullptr);

  LOCK_STATE();
  g_ctx.sensors[i].cfg_ok = multi_ok;
  memcpy(g_ctx.sensors[i].fw, fw, sizeof(fw));
  UNLOCK_STATE();
  snprintf(msg, sizeof(msg), "radar[%d] uart%u mount%+d: %s fw=%s multi=%s", i,
           RADAR_SENSORS[i].uart, (int)RADAR_SENSORS[i].mount_deg,
           multi_ok ? "OK" : "FAIL", fw[0] ? fw : "?",
           multi_ok ? "ok" : "no-ack");
  emit_log(multi_ok ? "info" : "warn", msg);
}

void radar_uart_init() {
  int up = 0;
  for (int i = 0; i < RADAR_SENSOR_COUNT; i++) {
    ld2450_parser_init(&s_parser[i]);
    s_port[i] = new HardwareSerial(RADAR_SENSORS[i].uart);
    s_port[i]->setRxBufferSize(RADAR_UART_RX_BUF);
    s_port[i]->begin(RADAR_UART_BAUD, SERIAL_8N1,
                     RADAR_SENSORS[i].rx_pin, RADAR_SENSORS[i].tx_pin);
  }
#if RADAR_SENSOR_BOOT_CONFIG
  // Modules take a moment after power-on before they answer config commands.
  delay(300);
  for (int i = 0; i < RADAR_SENSOR_COUNT; i++) boot_config(i);
#endif
  LOCK_STATE();
  for (int i = 0; i < RADAR_SENSOR_COUNT; i++) up += g_ctx.sensors[i].cfg_ok ? 1 : 0;
  UNLOCK_STATE();
  char msg[64];
  snprintf(msg, sizeof(msg), "radar: %d/%d sensors configured", up, RADAR_SENSOR_COUNT);
  emit_log(up == RADAR_SENSOR_COUNT ? "info" : "warn", msg);
}

void radar_uart_pump() {
  for (int i = 0; i < RADAR_SENSOR_COUNT; i++) {
    Ld2450Frame f;
    while (s_port[i]->available() > 0) {
      if (ld2450_feed(&s_parser[i], (uint8_t)s_port[i]->read(), &f)) {
        publish_frame(i, f);
      }
    }
  }
}

#else
// ===========================================================================
// STUB — no modules wired (they arrive tomorrow). A scripted scene is encoded
// into real LD2450 wire bytes and fed through the SAME parsers, so the whole
// parse → rotate → fuse → emit pipeline runs on a bare S3. Scene: one person
// orbiting the robot (crossing every mount seam ~every 40 s) plus a second
// person who steps in at -90° for a few seconds each cycle.
// ===========================================================================

static uint32_t s_last_tick_ms = 0;

// Robot-frame scene target -> this sensor's local slot, official wire
// convention (+x = right of sensor; the encoder writes the sign-flag format).
static bool scene_to_slot(float bearing_deg, float range_m, float speed_mps,
                          int sensor, Ld2450Target* out) {
  const float local = radar_wrap180(bearing_deg - RADAR_SENSORS[sensor].mount_deg);
  if (fabsf(local) > 60.0f) return false;   // outside this sensor's FOV
  const float rad = local * 0.017453292519943295f;
  out->x_mm      = (int16_t)(-range_m * 1000.0f * sinf(rad));
  out->y_mm      = (int16_t)(range_m * 1000.0f * cosf(rad));
  out->speed_cms = (int16_t)(speed_mps * 100.0f);
  out->res_mm    = 360;
  out->present   = true;
  return true;
}

void radar_uart_init() {
  for (int i = 0; i < RADAR_SENSOR_COUNT; i++) ld2450_parser_init(&s_parser[i]);
  s_last_tick_ms = millis();
  emit_log("info", "radar: STUB build — synthetic orbiting scene, no modules read");
}

void radar_uart_pump() {
  // The real modules free-run at 10 Hz; the stub matches.
  const uint32_t now = millis();
  if ((uint32_t)(now - s_last_tick_ms) < 100) return;
  s_last_tick_ms = now;

  const float t = (float)now * 0.001f;
  // Person A: full orbit every 40 s, breathing range 1.5..4 m.
  const float a_bearing = radar_wrap180(9.0f * t);
  const float a_range   = 2.75f + 1.25f * sinf(0.31f * t);
  const float a_speed   = 1.25f * 0.31f * cosf(0.31f * t);   // d(range)/dt
  // Person B: at -90° for 6 s out of every 20 s.
  const bool  b_here    = fmodf(t, 20.0f) < 6.0f;

  for (int i = 0; i < RADAR_SENSOR_COUNT; i++) {
    Ld2450Frame f = {};
    int slot = 0;
    Ld2450Target tgt;
    if (scene_to_slot(a_bearing, a_range, a_speed, i, &tgt)) f.t[slot++] = tgt;
    if (b_here && scene_to_slot(-90.0f, 1.8f, 0.0f, i, &tgt)) f.t[slot++] = tgt;

    uint8_t wire[LD2450_FRAME_BYTES];
    ld2450_encode_frame(&f, wire);
    Ld2450Frame parsed;
    for (int k = 0; k < LD2450_FRAME_BYTES; k++) {
      if (ld2450_feed(&s_parser[i], wire[k], &parsed)) publish_frame(i, parsed);
    }
  }
}
#endif
