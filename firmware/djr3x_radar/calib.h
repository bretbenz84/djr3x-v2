// calib.h — tuning constants for the LD2450 radar ring.
//
// The one file for thresholds and falloffs, mirroring the drive base's calib.h
// role. Everything here is a first-principles guess until the modules arrive;
// the bring-up pass (mutual interference, PETG penetration, seam behavior)
// tunes these against real frames. Header is Arduino-free on purpose: the pure
// parser/fusion modules (ld2450.cpp, fusion.cpp) compile on the HOST for the
// regression harness (tests/test_radar_parser.py), and they include this file.
#pragma once
#include "pins.h"   // RADAR_SENSOR_COUNT (Arduino-free, host-safe)

// ---- Sensor UART ----------------------------------------------------------
#define RADAR_UART_BAUD       256000   // LD2450 fixed default (protocol doc §1.2.1)
#define RADAR_UART_RX_BUF     512      // frames are 30 B at 10 Hz; generous

// ---- Freshness ------------------------------------------------------------
// A sensor's latest frame older than this is excluded from fusion and reported
// ok:false in telemetry. 300 ms = 3 missed frames at the module's 10 Hz — a
// genuinely wedged/unplugged sensor, not line jitter. (The "person froze and
// fell off the track list" latch is the MAC's job, seconds-scale, on the fused
// output — this window is transport health only.)
#define RADAR_SENSOR_STALE_MS 300

// ---- Geometry -------------------------------------------------------------
// The official protocol doc never states which side of the sensor is +x, and
// two of the three open-source drivers surveyed flip it (see the parser notes
// in ld2450.h). We implement the official + ESPHome-core convention (+x =
// right of sensor); if bring-up shows bearings mirrored (walk left, bearing
// goes right), set this to 1 rather than touching the math.
//
// RESOLVED 2026-08-14 on real hardware: bearings came out mirrored left/right,
// so these modules use the OPPOSITE x polarity from the official doc's worked
// example — i.e. the csRon / TillFleisch drivers had it right for this batch.
// Flag flipped to 1; the math in fusion.cpp is untouched and still matches the
// official doc, which is where it should stay.
#define RADAR_FLIP_X          1
#define RADAR_RANGE_MIN_M     0.20f    // closer than this is the robot's own shell/PETG
#define RADAR_RANGE_MAX_M     8.00f    // module spec ceiling; beyond is noise

// ---- Confidence falloff ---------------------------------------------------
// LD2450 angle accuracy degrades from ~2° at boresight toward ~20° at the ±60°
// FOV edges (module spec), so confidence is 1.0 inside ±RADAR_CONF_FULL_DEG,
// falls linearly to RADAR_CONF_EDGE_MIN at ±RADAR_CONF_EDGE_DEG, and is
// RADAR_CONF_BEYOND_EDGE past that (the module sometimes reports slightly
// outside its rated FOV; don't trust it, don't drop it).
#define RADAR_CONF_FULL_DEG    45.0f
#define RADAR_CONF_EDGE_DEG    60.0f
#define RADAR_CONF_EDGE_MIN    0.35f
#define RADAR_CONF_BEYOND_EDGE 0.20f

// ---- Seam dedup -----------------------------------------------------------
// A person near a 120° mount boundary is reported by BOTH adjacent sensors.
// Robot-frame targets within both thresholds merge into one (confidence-
// weighted; agreement RAISES confidence — see fusion.cpp). Bearing threshold
// covers the worst-case ±20° edge accuracy of two sensors disagreeing; range
// threshold covers the module's 0.75 m range resolution.
#define RADAR_DEDUP_BEARING_DEG 15.0f
#define RADAR_DEDUP_RANGE_M     0.80f

// ---- Capacities -----------------------------------------------------------
#define RADAR_MAX_RAW_TARGETS  (RADAR_SENSOR_COUNT * 3)  // 3 slots per module
#define RADAR_FUSED_MAX        6                          // fused list ceiling

// ---- Emit rate ------------------------------------------------------------
#define RADAR_TELEM_PERIOD_MS  100     // 10 Hz, matching the sensors' frame rate

// ---- Boot-time sensor config (real HW only) -------------------------------
// One config-mode transaction per sensor at init: read the firmware version
// into the logs and force MULTI-target tracking (a module left in single-target
// mode would silently cap the ring at 1 person). Read-only otherwise — no
// Bluetooth changes, no zone filters, nothing persistent.
#define RADAR_SENSOR_BOOT_CONFIG 1
#define RADAR_CFG_ACK_TIMEOUT_MS 300   // per command; a silent sensor skips the rest
