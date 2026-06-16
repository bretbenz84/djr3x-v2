// djr3x_motion.ino — DJ-R3X motion controller (ESP32) — Phase 0 bring-up.
//
// Implements the full Mac<->ESP32 wire protocol (docs/motion_protocol.md v1)
// against a STUBBED hardware layer: no motors, encoders, or ToF sensors need to
// be wired. The plant model in control.cpp synthesizes odometry from commanded
// velocity, so turn/move/come run to completion and emit `done`; ToF reads
// "clear" so the reflex/zone logic stays green. This lets the whole protocol be
// validated on a bare board, and lets the Mac side be developed against a real
// device. As peripherals are wired, flip MOTION_HW_PRESENT (hal.h) to 1 and
// fill the marked driver sections — nothing above the HAL changes.
//
// Build:  arduino-cli compile --fqbn esp32:esp32:esp32 firmware/djr3x_motion
// Upload: arduino-cli upload  --fqbn esp32:esp32:esp32 -p /dev/cu.usbserial-10 firmware/djr3x_motion
// See firmware/djr3x_motion/README.md for the manual protocol test recipe.

#include <Arduino.h>
#include <esp_random.h>
#include "context.h"
#include "protocol.h"
#include "hal.h"
#include "proto_io.h"
#include "control.h"
#include "safety.h"
#include "gamepad.h"

// ---- Globals (declared extern in context.h) ------------------------------
MotionContext     g_ctx;
SemaphoreHandle_t g_state_mux = nullptr;
SemaphoreHandle_t g_tx_mux    = nullptr;

// ---- FreeRTOS tasks ------------------------------------------------------
// Real-time control + serial pinned to core 1; sensor + telemetry on core 0.

static void controlTask(void*) {
  TickType_t last = xTaskGetTickCount();
  const float dt = 0.01f;                 // 100 Hz
  for (;;) {
    safety_tick();                        // reflexes first (may set BLOCKED/comms_lost)
    control_tick(dt);                     // then drive the plant + finite cmds
    vTaskDelayUntil(&last, pdMS_TO_TICKS(10));
  }
}

static void serialTask(void*) {
  for (;;) {
    proto_poll();                         // drain RX, frame + dispatch lines
    vTaskDelay(pdMS_TO_TICKS(2));         // ~500 Hz poll
  }
}

static void sensorTask(void*) {
  TickType_t last = xTaskGetTickCount();
  for (;;) {
    TofMm t;
    hal_read_tof(t);                      // stub: all clear
    LOCK_STATE(); g_ctx.tof = t; UNLOCK_STATE();
    vTaskDelayUntil(&last, pdMS_TO_TICKS(20));   // 50 Hz
  }
}

static void telemetryTask(void*) {
  TickType_t last = xTaskGetTickCount();
  for (;;) {
    emit_telemetry();
    vTaskDelayUntil(&last, pdMS_TO_TICKS(50));    // 20 Hz
  }
}

void setup() {
  Serial.begin(115200);
  delay(50);

  g_state_mux = xSemaphoreCreateRecursiveMutex();
  g_tx_mux    = xSemaphoreCreateMutex();

  g_ctx.boot_id     = (uint32_t)esp_random();
  g_ctx.last_mac_ms = millis();           // grace before the watchdog can arm
  g_ctx.seen_mac    = false;              // watchdog stays disarmed until first Mac line

  hal_init();
  hal_tof_init();
  proto_init();
  control_init();
  safety_init();
  gamepad_init();

  emit_event_boot(g_ctx.boot_id);         // announce reset (carries boot_id + fw)

  // Highest priority = control; serial just under it (both core 1).
  xTaskCreatePinnedToCore(controlTask,   "control", 4096, nullptr, 4, nullptr, 1);
  xTaskCreatePinnedToCore(serialTask,    "serial",  4096, nullptr, 3, nullptr, 1);
  xTaskCreatePinnedToCore(sensorTask,    "sensor",  3072, nullptr, 2, nullptr, 0);
  xTaskCreatePinnedToCore(telemetryTask, "telem",   4096, nullptr, 2, nullptr, 0);
}

void loop() {
  // The control/serial/sensor/telemetry work runs in the tasks above. The Arduino
  // loopTask polls the Bluetooth gamepad here (BP32.update needs frequent calls); with
  // the gamepad feature off, gamepad_tick() is a no-op and this idles slowly.
  gamepad_tick();
  vTaskDelay(pdMS_TO_TICKS(GAMEPAD_POLL_MS));
}
