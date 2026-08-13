// ld2450.cpp — HLK-LD2450 frame parser/encoder (pure; see ld2450.h for the
// cross-checked protocol facts and the sign-encoding trap).
#include "ld2450.h"
#include <string.h>

static const uint8_t DATA_HDR[4]  = {0xAA, 0xFF, 0x03, 0x00};
static const uint8_t DATA_TAIL[2] = {0x55, 0xCC};
static const uint8_t CMD_HDR[4]   = {0xFD, 0xFC, 0xFB, 0xFA};
static const uint8_t CMD_TAIL[4]  = {0x04, 0x03, 0x02, 0x01};

int16_t ld2450_decode_signed(uint16_t raw) {
  // Official: high bit 1 = POSITIVE (value = raw - 0x8000), 0 = NEGATIVE
  // (value = -raw). Sign-and-magnitude, not two's complement.
  return (raw & 0x8000) ? (int16_t)(raw - 0x8000) : (int16_t)(-(int32_t)raw);
}

uint16_t ld2450_encode_signed(int16_t v) {
  return v >= 0 ? (uint16_t)((uint16_t)v | 0x8000u) : (uint16_t)(-(int32_t)v);
}

static uint16_t le16(const uint8_t* p) {
  return (uint16_t)(p[0] | ((uint16_t)p[1] << 8));
}

static void put_le16(uint8_t* p, uint16_t v) {
  p[0] = (uint8_t)(v & 0xFF);
  p[1] = (uint8_t)(v >> 8);
}

void ld2450_parser_init(Ld2450Parser* p) {
  memset(p, 0, sizeof(*p));
}

static void decode_frame(const uint8_t* buf, Ld2450Frame* out) {
  for (int i = 0; i < LD2450_TARGET_SLOTS; i++) {
    const uint8_t* s = buf + 4 + 8 * i;
    Ld2450Target& t = out->t[i];
    bool all_zero = true;
    for (int k = 0; k < 8; k++) {
      if (s[k] != 0) { all_zero = false; break; }
    }
    if (all_zero) {
      t = Ld2450Target{0, 0, 0, 0, false};
      continue;
    }
    t.x_mm      = ld2450_decode_signed(le16(s + 0));
    t.y_mm      = ld2450_decode_signed(le16(s + 2));
    t.speed_cms = ld2450_decode_signed(le16(s + 4));
    t.res_mm    = le16(s + 6);
    t.present   = true;
  }
}

// Shift the buffer to the next plausible header start after a tail mismatch —
// a partial header at the end of the buffer is kept (it may complete).
static void resync(Ld2450Parser* p) {
  size_t shift = 1;
  while (shift < LD2450_FRAME_BYTES) {
    size_t avail = LD2450_FRAME_BYTES - shift;
    size_t need  = avail < 4 ? avail : 4;
    if (memcmp(p->buf + shift, DATA_HDR, need) == 0) break;
    shift++;
  }
  p->bytes_dropped += (uint32_t)shift;
  memmove(p->buf, p->buf + shift, LD2450_FRAME_BYTES - shift);
  p->len = (uint8_t)(LD2450_FRAME_BYTES - shift);
}

bool ld2450_feed(Ld2450Parser* p, uint8_t b, Ld2450Frame* out) {
  if (p->len < 4) {
    // Header hunt: a mismatched byte is dropped, but a byte that could START a
    // header is kept (handles "AA AA FF 03 00" run-ins).
    if (b == DATA_HDR[p->len]) {
      p->buf[p->len++] = b;
    } else if (b == DATA_HDR[0]) {
      p->bytes_dropped += p->len;
      p->buf[0] = b;
      p->len = 1;
    } else {
      p->bytes_dropped += (uint32_t)p->len + 1;
      p->len = 0;
    }
    return false;
  }
  p->buf[p->len++] = b;
  if (p->len < LD2450_FRAME_BYTES) return false;

  if (p->buf[28] == DATA_TAIL[0] && p->buf[29] == DATA_TAIL[1]) {
    decode_frame(p->buf, out);
    p->frames_ok++;
    p->len = 0;
    return true;
  }
  p->frames_bad++;
  resync(p);
  return false;
}

void ld2450_encode_frame(const Ld2450Frame* f, uint8_t out[LD2450_FRAME_BYTES]) {
  memcpy(out, DATA_HDR, 4);
  for (int i = 0; i < LD2450_TARGET_SLOTS; i++) {
    uint8_t* s = out + 4 + 8 * i;
    const Ld2450Target& t = f->t[i];
    if (!t.present) {
      memset(s, 0, 8);
      continue;
    }
    put_le16(s + 0, ld2450_encode_signed(t.x_mm));
    put_le16(s + 2, ld2450_encode_signed(t.y_mm));
    put_le16(s + 4, ld2450_encode_signed(t.speed_cms));
    put_le16(s + 6, t.res_mm);
  }
  memcpy(out + 28, DATA_TAIL, 2);
}

size_t ld2450_build_cmd(uint16_t word, const uint8_t* value, size_t value_len,
                        uint8_t* out) {
  // len field counts word + value only (official doc §2.1.2).
  size_t n = 0;
  memcpy(out, CMD_HDR, 4);              n += 4;
  put_le16(out + n, (uint16_t)(2 + value_len)); n += 2;
  put_le16(out + n, word);              n += 2;
  if (value_len > 0) { memcpy(out + n, value, value_len); n += value_len; }
  memcpy(out + n, CMD_TAIL, 4);         n += 4;
  return n;
}

bool ld2450_find_ack(const uint8_t* buf, size_t n, uint16_t word,
                     const uint8_t** value, size_t* value_len) {
  const uint16_t ack_word = (uint16_t)(word | 0x0100);
  for (size_t i = 0; i + 12 <= n; i++) {
    if (memcmp(buf + i, CMD_HDR, 4) != 0) continue;
    const uint16_t len = le16(buf + i + 4);           // word + value bytes
    if (len < 2 || len > 64) continue;                // implausible — keep scanning
    const size_t end = i + 6 + len;                   // start of tail
    if (end + 4 > n) continue;                        // truncated — not this one
    if (memcmp(buf + end, CMD_TAIL, 4) != 0) continue;
    if (le16(buf + i + 6) != ack_word) continue;
    *value = buf + i + 8;
    *value_len = (size_t)(len - 2);
    return true;
  }
  return false;
}
