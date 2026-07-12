// gamepad.h — optional Bluetooth gamepad manual override (Bluepad32), docs §11.
//
// MOTION_GAMEPAD_PRESENT gates the whole feature. While it is 0 (the default) both
// hooks below are no-ops and the firmware has NO Bluetooth/Bluepad32 dependency — it
// builds on the plain ESP32 core like always. Build with -DMOTION_GAMEPAD_PRESENT=1
// to compile the real driver; THAT build requires the Bluepad32 board package (a
// BTstack-based ESP32 core), not the stock esp32:esp32 core — see README "Manual
// gamepad override".
//
// The gamepad pairs directly to the ESP32 (not the Mac), so it overrides autonomous/
// voice motion and keeps working even with the USB link down. The arbitration it
// drives (owner=MANUAL, ToF full-override, disconnect failsafe) lives in control.cpp /
// safety.cpp and is always compiled — only the Bluepad32 I/O is behind the flag.
#pragma once

#ifndef MOTION_GAMEPAD_PRESENT
#define MOTION_GAMEPAD_PRESENT 0   // override per-build: -DMOTION_GAMEPAD_PRESENT=1
#endif

// Arduino loop() poll cadence. The gamepad needs frequent BP32.update() calls; with
// the feature off there's nothing to poll, so idle slowly.
#if MOTION_GAMEPAD_PRESENT
#define GAMEPAD_POLL_MS 15
#else
#define GAMEPAD_POLL_MS 1000
#endif

void gamepad_init();   // start Bluepad32 (no-op in the stub)
void gamepad_tick();   // poll the pad + drive manual arbitration (no-op in the stub)
// A host (main.py / bench tool) just completed the `hello` handshake — greet the
// operator with a rumble double-pulse. Callable from ANY task (sets a flag; the
// actual Bluepad32 call happens in gamepad_tick on the loopTask). No-op in the stub.
void gamepad_notify_host_connected();
