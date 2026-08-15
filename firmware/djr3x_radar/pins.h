// pins.h — ESP32-S3 (N16R8) UART assignments for the LD2450 radar ring.
//
// The Mac link is NATIVE USB CDC (the S3's USB-OTG/JTAG port), so all three
// hardware UARTs (0/1/2) are free for sensors — no software serial anywhere.
// Pin choices avoid the S3 footguns:
//   - GPIO 26..32 are wired to the SPI flash. Never usable.
//   - GPIO 33..37 are bonded to the in-package OCTAL PSRAM on N16R8 modules
//     (this board). Unusable here regardless of whether the app enables PSRAM.
//   - GPIO 19/20 are the native USB D-/D+ — the Mac link itself. Reserved.
//   - GPIO 0/3/45/46 are strapping pins; kept off the sensor lines.
//   - GPIO 43/44 are the UART0 default pins (the "COM" header / bridge chip on
//     dual-port devkits). We re-map UART0 onto clean GPIOs instead so the
//     bridge port stays free for debugging.
// That leaves 4-18, 21, 38-42, 47, 48 clean; sensors use the low block.
//
// If your wiring differs, THIS is the one file to edit — nothing else
// hard-codes a pin or a mount angle.
#pragma once
#include <stdint.h>

// One row per LD2450 module. Written for N sensors, not hardcoded 3 — during
// bring-up you can run 2 by deleting a row and dropping RADAR_SENSOR_COUNT;
// everything (parsers, fusion, telemetry, boot config) iterates the table.
//
//   uart       ESP32-S3 UART controller (0/1/2), one per sensor.
//   rx_pin     ESP GPIO wired to the sensor's TX (the data stream — required).
//   tx_pin     ESP GPIO wired to the sensor's RX. Wired even though reading
//              needs only RX, so config can be pushed without pulling a module
//              out of the ring (spec "Notes").
//   mount_deg  Where the sensor's boresight points in the ROBOT frame, in the
//              project-wide sign convention (docs/motion_protocol.md §4):
//              degrees, 0 = robot forward, + = LEFT/CCW, wrapped (-180,180].
struct RadarSensorPin {
  uint8_t uart;
  int8_t  tx_pin;
  int8_t  rx_pin;
  float   mount_deg;
};

#define RADAR_SENSOR_COUNT 3

// Ring layout (2026-08-15): TWO FORWARD-QUARTER modules + ONE REAR. Mounts stay
// 120° apart, so the ring is the original 0°/±120° arrangement turned 60°: the
// pair straddles the front and the lone module points dead astern. With ±60°
// FOV per module the seams (where two sensors overlap at their FOV edges and
// fusion.cpp dedups) now fall at 0° — DEAD AHEAD — and at ±120°; the rear
// module's boresight sits on the ±180° wrap. A person straight in front is at
// BOTH forward modules' 60° edge, so his bearing there is the two-sensor merge,
// not a boresight read (the camera owns the front anyway — the radar is a
// prior). If your modules sit in different slots, edit ONLY the mount_deg
// column: the UART/pin columns describe the harness, not the ring.
static const RadarSensorPin RADAR_SENSORS[RADAR_SENSOR_COUNT] = {
  {1, 5,  4,   60.0f},   // S0 — front-left  (+60° CCW)
  {2, 7,  6,  -60.0f},   // S1 — front-right (60° CW)
  {0, 9,  8,  180.0f},   // S2 — rear (dead astern; +180 is the wrapped value)
};
