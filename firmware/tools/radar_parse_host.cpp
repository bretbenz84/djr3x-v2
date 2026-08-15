// radar_parse_host.cpp — host-side test shim for the radar firmware's pure
// parser/fusion modules (NOT firmware; never flashed).
//
// tests/test_radar_parser.py compiles this with clang++ against the ACTUAL
// firmware sources (firmware/djr3x_radar/ld2450.cpp + fusion.cpp — both
// Arduino-free by contract) and drives it with synthetic LD2450 byte streams,
// so the regression harness exercises the same code the ESP32-S3 runs, with no
// hardware attached. Lives in firmware/tools/ (a sibling of the sketch dir) so
// the Arduino build never sees a second main().
//
// Modes:
//   radar_parse_host parse
//     stdin: raw bytes. For each completed frame prints one JSON line
//     {"targets":[{"x":..,"y":..,"speed":..,"res":..},..]} (present slots
//     only); at EOF prints {"summary":{"frames_ok":..,"frames_bad":..,
//     "bytes_dropped":..}}.
//   radar_parse_host fuse [--flip]
//     stdin: one target per line, "sensor mount_deg x_mm y_mm speed_cms";
//     a blank line ends a tick -> prints one JSON line
//     {"fused":[{"b":..,"r":..,"c":..,"s":..,"m":..}]} for it.
//   radar_parse_host build <word_hex> [value_hex]
//     prints {"cmd":"<hex>"} — the config frame ld2450_build_cmd() would send.
//   radar_parse_host ack <word_hex>
//     stdin: raw bytes (ACKs, optionally with data frames interleaved).
//     prints {"found":true,"value":"<hex>"} or {"found":false}.
#include "../djr3x_radar/ld2450.h"
#include "../djr3x_radar/fusion.h"
#include "../djr3x_radar/calib.h"
#include <cstdio>
#include <cstring>
#include <cstdlib>

static int run_parse() {
  Ld2450Parser p;
  ld2450_parser_init(&p);
  Ld2450Frame f;
  int c;
  while ((c = getchar()) != EOF) {
    if (!ld2450_feed(&p, (uint8_t)c, &f)) continue;
    printf("{\"targets\":[");
    bool first = true;
    for (int i = 0; i < LD2450_TARGET_SLOTS; i++) {
      if (!f.t[i].present) continue;
      printf("%s{\"x\":%d,\"y\":%d,\"speed\":%d,\"res\":%u}", first ? "" : ",",
             f.t[i].x_mm, f.t[i].y_mm, f.t[i].speed_cms, f.t[i].res_mm);
      first = false;
    }
    printf("]}\n");
  }
  printf("{\"summary\":{\"frames_ok\":%u,\"frames_bad\":%u,\"bytes_dropped\":%u}}\n",
         p.frames_ok, p.frames_bad, p.bytes_dropped);
  return 0;
}

static void flush_tick(RadarTargetRobot* raw, int n_raw) {
  RadarTargetRobot fused[RADAR_FUSED_MAX];
  const int n = radar_fuse(raw, n_raw, fused, RADAR_FUSED_MAX);
  printf("{\"fused\":[");
  for (int i = 0; i < n; i++) {
    printf("%s{\"b\":%.2f,\"r\":%.3f,\"c\":%.3f,\"s\":%.2f,\"m\":%u}",
           i ? "" : "", fused[i].bearing_deg, fused[i].range_m,
           fused[i].confidence, fused[i].speed_mps, fused[i].sensors);
    if (i + 1 < n) printf(",");
  }
  printf("]}\n");
}

static int run_fuse(bool flip) {
  char line[128];
  RadarTargetRobot raw[RADAR_MAX_RAW_TARGETS];
  int n_raw = 0;
  bool pending = false;   // target lines read since the last flush — even if
                          // every one was discarded, that tick must still be
                          // reported (as an empty fused list), not swallowed
  while (fgets(line, sizeof(line), stdin)) {
    int sensor, x, y, speed;
    float mount;
    if (line[0] == '\n' || line[0] == '\r') {
      flush_tick(raw, n_raw);
      n_raw = 0;
      pending = false;
      continue;
    }
    if (sscanf(line, "%d %f %d %d %d", &sensor, &mount, &x, &y, &speed) != 5) {
      fprintf(stderr, "fuse: bad line: %s", line);
      return 2;
    }
    pending = true;
    Ld2450Target t;
    t.x_mm = (int16_t)x; t.y_mm = (int16_t)y; t.speed_cms = (int16_t)speed;
    t.res_mm = 360; t.present = true;
    RadarTargetRobot r;
    if (radar_local_to_robot(&t, sensor, mount, flip, &r) &&
        n_raw < RADAR_MAX_RAW_TARGETS) {
      raw[n_raw++] = r;
    }
  }
  if (pending) flush_tick(raw, n_raw);   // unterminated final tick
  return 0;
}

// "01ff" -> {0x01, 0xff}. Returns the byte count, or -1 on a malformed string.
static int unhex(const char* s, uint8_t* out, size_t cap) {
  const size_t len = strlen(s);
  if (len % 2 || len / 2 > cap) return -1;
  for (size_t i = 0; i < len; i += 2) {
    char byte[3] = {s[i], s[i + 1], 0};
    char* end = nullptr;
    const long v = strtol(byte, &end, 16);
    if (end != byte + 2) return -1;
    out[i / 2] = (uint8_t)v;
  }
  return (int)(len / 2);
}

static void print_hex(const uint8_t* b, size_t n) {
  for (size_t i = 0; i < n; i++) printf("%02x", b[i]);
}

static int run_build(const char* word_hex, const char* value_hex) {
  uint8_t wbuf[2], value[64];
  if (unhex(word_hex, wbuf, sizeof(wbuf)) != 2) {
    fprintf(stderr, "build: word must be 2 hex bytes\n");
    return 2;
  }
  const uint16_t word = (uint16_t)((wbuf[0] << 8) | wbuf[1]);   // 00a5 -> 0x00A5
  int vlen = 0;
  if (value_hex && (vlen = unhex(value_hex, value, sizeof(value))) < 0) {
    fprintf(stderr, "build: bad value hex\n");
    return 2;
  }
  uint8_t out[128];
  const size_t n = ld2450_build_cmd(word, vlen ? value : nullptr, (size_t)vlen, out);
  printf("{\"cmd\":\"");
  print_hex(out, n);
  printf("\"}\n");
  return 0;
}

static int run_ack(const char* word_hex) {
  uint8_t wbuf[2];
  if (unhex(word_hex, wbuf, sizeof(wbuf)) != 2) {
    fprintf(stderr, "ack: word must be 2 hex bytes\n");
    return 2;
  }
  const uint16_t word = (uint16_t)((wbuf[0] << 8) | wbuf[1]);
  uint8_t buf[512];
  size_t n = 0;
  int c;
  while ((c = getchar()) != EOF && n < sizeof(buf)) buf[n++] = (uint8_t)c;
  const uint8_t* value; size_t vlen;
  if (!ld2450_find_ack(buf, n, word, &value, &vlen)) {
    printf("{\"found\":false}\n");
    return 0;
  }
  printf("{\"found\":true,\"value\":\"");
  print_hex(value, vlen);
  printf("\"}\n");
  return 0;
}

int main(int argc, char** argv) {
  if (argc >= 2 && !strcmp(argv[1], "parse")) return run_parse();
  if (argc >= 2 && !strcmp(argv[1], "fuse")) {
    const bool flip = argc >= 3 && !strcmp(argv[2], "--flip");
    return run_fuse(flip);
  }
  if (argc >= 3 && !strcmp(argv[1], "build")) {
    return run_build(argv[2], argc >= 4 ? argv[3] : nullptr);
  }
  if (argc >= 3 && !strcmp(argv[1], "ack")) return run_ack(argv[2]);
  fprintf(stderr, "usage: %s parse|fuse [--flip]|build <word> [value]|ack <word>\n",
          argv[0]);
  return 2;
}
