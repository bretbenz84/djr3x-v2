// tof.cpp — ToF (Time-of-Flight) distance subsystem: 8 radial sensors.
//
// Layout (docs/motion_protocol.md §6): mounted at the 540 mm base-ring surface, every
// 45° starting 22.5° off the forward axis (no sensor on the cardinals themselves):
//   - 2× LONG  VL53L1X FRONT pair (fl / fr, ±22.5° off forward)  — wall sense + stop reflex
//   - 2× LONG  VL53L1X REAR  pair (rl / rr, ±22.5° off rearward) — reversing reflex
//   - 2× SHORT VL53L0X LEFT  pair (lf / lb, ±22.5° off left)     — lateral clearance
//   - 2× SHORT VL53L0X RIGHT pair (rf / rb, ±22.5° off right)    — lateral clearance
// The long pairs give room-scale range fore/aft; the short pairs read the hallway
// wall distance for the manual steering assist (control.cpp).
//
// Owns hal_read_tof() / hal_tof_init() for ALL builds, gated by MOTION_TOF_PRESENT
// (hal.h) independently of the motor drivers — the base can drive on real motors +
// encoders while the ToF sensors are still unwired. Until they are wired, the stub
// reports a clear room (obstacle avoidance inactive); safety.cpp's zone reflex runs
// against whatever this returns.
//
// ⚠ SCAFFOLD — validate on hardware. The addressing sequence, timing budgets, and the
//   sensor→field mapping all need a bench check. Bring-up aid: hal_tof_init() emits one
//   `log` line per sensor (OK/FAIL) + a tally over the wire, so wiring the bus is
//   observable in the serial monitor / Mac logs ([motion_fw] tof[…]) not silent.
//
// Addressing: TCA9548A I²C mux ONLY. All 8 sensors keep the default 0x29 and the mux
// selects one channel at a time (zero XSHUT GPIOs). 8 sensors exceed the ESP32's free
// GPIOs for XSHUT sequencing, so that scheme is unsupported for this layout (#error).
//
// FRONT MATRIX OVERLAY (MOTION_TOF_MATRIX_PRESENT, tof_matrix.cpp): the DFRobot 8x8
// Matrix ToF at the direct front contributes floor-rejected left/right-half obstacle
// distances that override (radial absent) or min-combine into (radial present) the
// front pair fl/fr at the end of hal_read_tof — the zone reflex sees ONE front pair
// either way. Rear/side fields are untouched by the matrix.
#include "hal.h"            // MOTION_TOF_PRESENT / _TOF_MATRIX_PRESENT / _TOF_USE_MUX + TofMm
#include "tof_matrix.h"     // front 8x8 matrix (no-op stubs when not built in)

#if MOTION_TOF_MATRIX_PRESENT
// Conservative front-pair combine for builds with BOTH the radial array and the
// matrix: nearest valid wins; -1 only when both are errored/absent.
static inline int16_t tof_front_combine(int16_t radial, int16_t matrix) {
  if (radial < 0) return matrix;
  if (matrix < 0) return radial;
  return (radial < matrix) ? radial : matrix;
}
#endif

#if MOTION_TOF_PRESENT
#include "pins.h"
#include "calib.h"
#include "tof_filter.h"
#include "proto_io.h"       // emit_log — per-sensor bring-up diagnostics
#include "i2c_trunk.h"
#include <Arduino.h>
#include <Wire.h>
#include <VL53L0X.h>        // Pololu vl53l0x-arduino (short-range diagonals)
#include <VL53L1X.h>        // Pololu vl53l1x-arduino (long-range cardinals)

#if !MOTION_TOF_USE_MUX
#error "The 8-sensor ToF layout (4x VL53L0X + 4x VL53L1X) requires the TCA9548A mux: \
8 sensors exceed the ESP32's free XSHUT GPIOs. Build with -DMOTION_TOF_USE_MUX=1 (default)."
#endif

// Index order == read order == TofMm field order. First TOF_SHORT_COUNT are the
// short VL53L0X (mux 0..3); the rest are the long VL53L1X (mux 4..7). The mux channel
// for index i is simply i. EDIT THIS TABLE (and hal_read_tof below) to match wiring.
//   idx mux type     field  placement (screen bearing; front = up = 90°, CCW+)
//   0   0   VL53L0X  lf     left-front   (157.5°)
//   1   1   VL53L0X  lb     left-back    (202.5°)
//   2   2   VL53L0X  rf     right-front  ( 22.5°)
//   3   3   VL53L0X  rb     right-back   (337.5°)
//   4   4   VL53L1X  fl     front-left   (112.5°)
//   5   5   VL53L1X  fr     front-right  ( 67.5°)
//   6   6   VL53L1X  rl     rear-left    (247.5°)
//   7   7   VL53L1X  rr     rear-right   (292.5°)
static const char* const TOF_LABEL[TOF_COUNT] = {
  "lf", "lb", "rf", "rb", "fl", "fr", "rl", "rr",
};

static VL53L0X s_short[TOF_SHORT_COUNT];   // index i in [0, TOF_SHORT_COUNT)
static VL53L1X s_long[TOF_LONG_COUNT];     // index i - TOF_SHORT_COUNT in [0, TOF_LONG_COUNT)
static bool    s_ok[TOF_COUNT] = {false};  // did this sensor init? gates reads (skip dead ones)

// Latest distance per sensor (persists across calls). hal_read_tof reads ONE sensor
// per call (round-robin) so a blocking continuous read never stalls the loop; the rest
// keep their last value. -1 = error / not present.
static int16_t s_dist[TOF_COUNT] = { -1, -1, -1, -1, -1, -1, -1, -1 };
static uint8_t s_err_streak[TOF_COUNT] = {0};   // consecutive failed reads per sensor
static int     s_next = 0;                  // round-robin cursor

// Per-sensor filter (tof_filter.h): fast-attack / slow-release + big-drop
// confirmation, so a single-frame phantom near return can't flap the reflex.
static TofFilt s_filt[TOF_COUNT];

static inline uint8_t mux_ch(int i) { return (uint8_t)i; }   // sensor index -> mux channel

// Select a mux channel; returns false if the mux did not ACK (so we don't trust a read
// of a possibly-still-old channel).
static bool mux_select(uint8_t ch) {
  Wire.beginTransmission(TOF_MUX_ADDR);
  Wire.write(1 << ch);
  return Wire.endTransmission() == 0;
}

// Final "N/8 up" tally after init — warn (not info) if any sensor is missing so a
// half-wired bus is obvious at a glance.
static void tof_report_tally() {
  int up = 0;
  for (int i = 0; i < TOF_COUNT; i++) up += s_ok[i] ? 1 : 0;
  char buf[48];
  snprintf(buf, sizeof(buf), "tof: %d/%d sensors up", up, TOF_COUNT);
  emit_log(up == TOF_COUNT ? "info" : "warn", buf);
}

void hal_tof_init() {
  // Wire lifecycle, 100 kHz clock, and bounded timeout are owned centrally by
  // i2c_trunk. No client may re-begin or silently retune the shared controller.

  // Probe the mux first: if it doesn't ACK, every sensor below will "fail" — say so
  // once, pointing at the real culprit (mux address / SDA-SCL / power) not 8 sensors.
  Wire.beginTransmission(TOF_MUX_ADDR);
  if (Wire.endTransmission() != 0) {
    char buf[64];
    snprintf(buf, sizeof(buf), "tof: TCA9548A mux not found at 0x%02X", TOF_MUX_ADDR);
    emit_log("error", buf);
  }

  // ---- Short-range VL53L0X on mux 0..3 ----
  for (int i = 0; i < TOF_SHORT_COUNT; i++) {
    mux_select(mux_ch(i));                       // only this sensor is on the bus now
    s_short[i].setTimeout(TOF_TIMEOUT_MS);
    s_ok[i] = s_short[i].init();                 // each keeps the default 0x29 behind the mux
    if (s_ok[i]) {
      s_short[i].setMeasurementTimingBudget(TOF_L0X_TIMING_BUDGET_US);
      s_short[i].startContinuous();
    }
    char buf[72];
    snprintf(buf, sizeof(buf), "tof[%d] %-5s ch%u VL53L0X: %s",
             i, TOF_LABEL[i], mux_ch(i), s_ok[i] ? "OK" : "FAIL");
    emit_log(s_ok[i] ? "info" : "warn", buf);
  }

  // ---- Long-range VL53L1X on mux 4..7 ----
  for (int j = 0; j < TOF_LONG_COUNT; j++) {
    const int i = TOF_SHORT_COUNT + j;
    mux_select(mux_ch(i));
    s_long[j].setTimeout(TOF_TIMEOUT_MS);
    s_ok[i] = s_long[j].init();
    if (s_ok[i]) {
      s_long[j].setDistanceMode(VL53L1X::Long);
      s_long[j].setMeasurementTimingBudget(TOF_L1X_TIMING_BUDGET_US);
      s_long[j].startContinuous(TOF_L1X_INTERMEASUREMENT_MS);
    }
    char buf[72];
    snprintf(buf, sizeof(buf), "tof[%d] %-5s ch%u VL53L1X: %s",
             i, TOF_LABEL[i], mux_ch(i), s_ok[i] ? "OK" : "FAIL");
    emit_log(s_ok[i] ? "info" : "warn", buf);
  }

  tof_report_tally();

#if MOTION_TOF_MATRIX_PRESENT
  tof_matrix_init();        // deferred background init (~5 s settle off this path)
#endif
}

// Read one sensor by index. Returns mm, or -1 on a genuine read error / no comms.
// A value at/over the per-type out-of-range cap means "nothing in range = clear".
static int read_mm(int i) {
  if (!s_ok[i]) return -1;                        // dead sensor: skip the blocking I²C wait
  if (!mux_select(mux_ch(i))) return -1;           // mux NACK: channel didn't switch — don't trust it

  if (i < TOF_SHORT_COUNT) {                       // VL53L0X
    const int mm = s_short[i].readRangeContinuousMillimeters();
    if (s_short[i].timeoutOccurred()) return -1;
    if (mm >= TOF_L0X_OUT_OF_RANGE_MM) return TOF_L0X_OUT_OF_RANGE_MM;
    return mm;
  }

  VL53L1X& s = s_long[i - TOF_SHORT_COUNT];        // VL53L1X
  const int mm = (int)s.read();                    // blocking read of the continuous result
  if (s.timeoutOccurred()) return -1;
  // Accept only plain valid + min-range-clipped. RangeValidNoWrapCheckFail is
  // REJECTED (treated as out-of-range): field data 2026-07-21 showed it is the
  // wrap-around phantom status — sensors staring into open space aliased far/no
  // returns into random near distances (fl/fr/rr sawtoothed 4 m -> 1 m in an
  // empty room, while rl with a real 1.9 m wall sat rock steady on RangeValid).
  // A real target passes the wrap check on the very next frame, so the cost is
  // one round-robin revisit (~80 ms) of detection latency.
  const VL53L1X::RangeStatus rs = s.ranging_data.range_status;
  const bool valid = (rs == VL53L1X::RangeValid ||
                      rs == VL53L1X::RangeValidMinRangeClipped);
  if (!valid || mm >= TOF_L1X_OUT_OF_RANGE_MM) return TOF_L1X_OUT_OF_RANGE_MM;
  return mm;
}

static void poll_one(int i) {
  if (!s_ok[i]) {
    s_dist[i] = -1;                               // not present -> honest error/no-data
    return;
  }
  const int mm = read_mm(i);
  if (mm >= 0) {
    s_err_streak[i] = 0;
    // tof_filter.h: nearer readings attack fast (a big one-frame drop needs a
    // confirming 2nd frame — anti-phantom), farther ones release at a bounded
    // rate. Kills both edge-of-beam strobing and single-frame speckle dips.
    s_dist[i] = tof_filter_step(s_filt[i], (int16_t)mm);
    return;
  }
  // Failed read: hold the last-good value through a TRANSIENT error, but not forever —
  // a sensor that dies while reading "clear" would otherwise freeze that clear distance
  // and silently disable the stop reflex in its direction. After a solid failure streak
  // publish an honest -1 (safety fails open on -1 by documented choice, and the GUI
  // radar/telemetry show the dead sensor instead of a healthy-looking stale distance).
  if (s_err_streak[i] < 255) s_err_streak[i]++;
  if (s_err_streak[i] == TOF_ERR_STREAK_STALE && s_dist[i] != -1) {
    s_dist[i] = -1;
    char buf[72];
    snprintf(buf, sizeof(buf), "tof[%d] %s: %u consecutive read errors - reporting -1",
             i, TOF_LABEL[i], (unsigned)TOF_ERR_STREAK_STALE);
    emit_log("warn", buf);
  } else if (s_err_streak[i] > TOF_ERR_STREAK_STALE) {
    s_dist[i] = -1;                               // stay honest until a read succeeds
  }
}

void hal_read_tof(TofMm& out) {
  // Round-robin, TWO sensors per call (one short + one long) so each sensor refreshes
  // every 4 task ticks (~80 ms at the 50 Hz sensor task) — fresh enough for the stop
  // reflex and the hallway assist at teleop speeds (≤ ~15 mm of travel per revisit).
  // Each continuous sensor produces a sample every ~33-60 ms, so by the time the
  // cursor revisits one its sample is ready and the blocking read returns immediately.
  const int s = s_next;                            // 0..3: shorts (mux 0..3)
  const int l = TOF_SHORT_COUNT + s_next;          // 4..7: longs  (mux 4..7)
  s_next = (s_next + 1) % TOF_SHORT_COUNT;
  poll_one(s);
  poll_one(l);

  out.lf = s_dist[0];   // short, mux 0 — left-front
  out.lb = s_dist[1];   // short, mux 1 — left-back
  out.rf = s_dist[2];   // short, mux 2 — right-front
  out.rb = s_dist[3];   // short, mux 3 — right-back
  out.fl = s_dist[4];   // long,  mux 4 — front-left
  out.fr = s_dist[5];   // long,  mux 5 — front-right
  out.rl = s_dist[6];   // long,  mux 6 — rear-left
  out.rr = s_dist[7];   // long,  mux 7 — rear-right

#if MOTION_TOF_MATRIX_PRESENT
  // Front matrix overlay: min-combine with the radial front pair (nearest wins).
  int16_t mfl, mfr;
  tof_matrix_read(&mfl, &mfr);
  out.fl = tof_front_combine(out.fl, mfl);
  out.fr = tof_front_combine(out.fr, mfr);
#endif
}

#else
// ===========================================================================
// STUB — no radial ToF sensors wired. Rear/side fields report a clear room.
// WITHOUT the matrix: obstacle avoidance is fully inactive in this build.
// WITH the matrix (MOTION_TOF_MATRIX_PRESENT=1): the front 8x8 OWNS fl/fr —
// including an honest -1 while initializing (~6 s after boot) or errored — so
// the FRONT stop/slow reflex is real; reversing remains unprotected.
// ===========================================================================
void hal_tof_init() {
#if MOTION_TOF_MATRIX_PRESENT
  tof_matrix_init();        // deferred background init (~5 s settle off this path)
#endif
}

void hal_read_tof(TofMm& out) {
  out.fl = out.fr = out.rl = out.rr = 4000;   // long pairs: room reads clear
  out.lf = out.lb = out.rf = out.rb = 1500;   // short pairs: no walls in range
#if MOTION_TOF_MATRIX_PRESENT
  // The matrix is the ONLY front sensor here: it owns fl/fr outright, so a dead
  // or still-initializing matrix shows as -1 in telemetry (visible in the GUI
  // radar) instead of hiding behind the stub's fictional 4000-clear.
  tof_matrix_read(&out.fl, &out.fr);
#endif
}
#endif  // MOTION_TOF_PRESENT
