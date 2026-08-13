// ld2450.h — HLK-LD2450 frame parser, encoder, and config-command builders.
//
// PURE module: stdint/string/stdbool only, no Arduino includes — it compiles
// on the HOST for the regression harness (tests/test_radar_parser.py builds it
// with clang++ and drives it with synthetic byte streams) and on the ESP32-S3
// unchanged. Keep it that way: no millis(), no Serial, no FreeRTOS.
//
// Protocol facts below were cross-checked 2026-08-12 against (a) Hi-Link's
// official "LD2450 Serial Communication Protocol" V1.03 PDF, (b) the Hi-Link
// operation manual's worked byte examples, and (c) the ESPHome core `ld2450`
// component source. DO NOT "fix" the sign decode from another driver: the
// csRon/HLK-LD2450 Python driver and the TillFleisch ESPHome fork both decode
// x/speed with the OPPOSITE polarity to the official doc's own worked example
// (csRon even yields negative Y, which the manual says cannot occur). Official
// convention, verified against the doc's example frame:
//
//   DATA frame (10 Hz, 256000 8N1): AA FF 03 00 | 3 slots x 8 bytes | 55 CC
//     = 30 bytes fixed, no checksum, little-endian. An absent target's slot is
//     all zeros ("up to 3 targets" is filtering, not variable length).
//   Slot: x(2) y(2) speed(2) resolution(2).
//     x, y in mm; speed in cm/s; resolution in mm (constant ~360, unused).
//     Sign encoding is sign-and-magnitude with an INVERTED flag — high bit 1
//     means POSITIVE: value = (raw & 0x8000) ? raw - 0x8000 : -raw.
//     Worked example from the doc: raw x=0x030E -> -782 mm; raw y=0x86B1 ->
//     +1713 mm; raw speed=0x0010 -> -16 cm/s.
//   y is the sensor boresight (always positive in practice); which side is +x
//     is NOT stated officially — ESPHome's docs say +x = right of sensor. The
//     RADAR_FLIP_X calib flag exists because two drivers disagree; verify once
//     against a live module (walk to one side) and set the flag, don't edit
//     the decode.
//
//   CONFIG frames (different framing): FD FC FB FA | len(2 LE) | word(2 LE) |
//     value | 04 03 02 01. ACK echoes word|0x0100; return value starts with a
//     2-byte status (0 = ok). Commands are only valid inside an enable-config
//     /end-config bracket, and data reporting PAUSES while config mode is
//     open. NOTE the second sign trap: zone-filter coordinates in the config
//     protocol are ordinary TWO'S COMPLEMENT int16 mm — the data-frame
//     sign-flag scheme applies to data frames only.
#pragma once
#include <stdint.h>
#include <stddef.h>

#define LD2450_FRAME_BYTES  30
#define LD2450_TARGET_SLOTS 3

// Config-command words (official doc §2.2; ACK word = these | 0x0100).
#define LD2450_CMD_ENABLE_CONFIG 0x00FF
#define LD2450_CMD_END_CONFIG    0x00FE
#define LD2450_CMD_MULTI_TARGET  0x0090
#define LD2450_CMD_READ_FW       0x00A0

struct Ld2450Target {
  int16_t  x_mm;      // + = right of sensor per official convention (see header)
  int16_t  y_mm;      // boresight distance; always positive for a real target
  int16_t  speed_cms; // radial; ESPHome maps + = moving away (unofficial)
  uint16_t res_mm;    // "distance resolution" — constant, unused
  bool     present;   // false = the slot was all zeros
};

struct Ld2450Frame {
  Ld2450Target t[LD2450_TARGET_SLOTS];
};

// Incremental byte-stream parser, one per sensor UART. Resyncs on garbage
// (boot chatter, mid-frame truncation, interleaved config ACKs) by hunting for
// the next header — a bad byte costs at most one frame, never wedges.
struct Ld2450Parser {
  uint8_t  buf[LD2450_FRAME_BYTES];
  uint8_t  len;
  uint32_t frames_ok;     // complete frames with a valid tail
  uint32_t frames_bad;    // full 30 bytes accumulated but tail mismatched
  uint32_t bytes_dropped; // bytes discarded while resyncing
};

void ld2450_parser_init(Ld2450Parser* p);

// Feed one byte; returns true when a complete valid frame was decoded into
// *out (parser state resets for the next frame).
bool ld2450_feed(Ld2450Parser* p, uint8_t b, Ld2450Frame* out);

// The official sign decode (see header). Exposed for tests.
int16_t  ld2450_decode_signed(uint16_t raw);
uint16_t ld2450_encode_signed(int16_t v);   // exact inverse, for the synth paths

// Encode a frame in the official wire format — used by the stub build's
// synthetic sensors and by host-side generators, so the same bytes the parser
// eats in tests are what a real module would send.
void ld2450_encode_frame(const Ld2450Frame* f, uint8_t out[LD2450_FRAME_BYTES]);

// Build a config-protocol command frame into out (>= 12 + value_len bytes).
// Returns the byte count. value may be NULL when value_len == 0.
size_t ld2450_build_cmd(uint16_t word, const uint8_t* value, size_t value_len,
                        uint8_t* out);

// Scan a raw byte buffer for a config ACK to `word` (i.e. word|0x0100 inside
// FD FC FB FA ... 04 03 02 01 framing, skipping any interleaved data-frame
// bytes). Returns true and points *value/*value_len at the return-value bytes
// (starting with the 2-byte status) when found.
bool ld2450_find_ack(const uint8_t* buf, size_t n, uint16_t word,
                     const uint8_t** value, size_t* value_len);
