// djr3x_radar.ino — DJ-R3X radar bearing-prior ring (ESP32-S3, 3x HLK-LD2450).
//
// Reads up to RADAR_SENSOR_COUNT LD2450 mmWave modules over hardware UARTs,
// rotates each sensor's targets into the robot frame, dedups across the mount
// seams, and streams a fused (bearing, range, confidence) list to the Mac at
// 10 Hz over native USB CDC — the coarse "start looking at 137°" hint for the
// come-here person search (docs/radar-bearing-prior-spec.md). This board is a
// SENSOR: it moves nothing, so there is no watchdog/estop machinery; it boots
// streaming and stays honest about what it can't see (radar.ok=false).
//
// Build (stub, bare board — the repo default, mirrors the drive base):
//   arduino-cli compile --fqbn esp32:esp32:esp32s3:CDCOnBoot=cdc firmware/djr3x_radar
// Build (real sensors wired):
//   arduino-cli compile --fqbn esp32:esp32:esp32s3:CDCOnBoot=cdc \
//     --build-property "compiler.cpp.extra_flags=-DRADAR_HW_PRESENT=1" \
//     firmware/djr3x_radar
// See firmware/djr3x_radar/README.md for flashing + the CDC-vanish recovery note.

#include <Arduino.h>
#include <esp_random.h>
#include "context.h"
#include "protocol.h"
#include "proto_io.h"
#include "radar_uart.h"

// ---- Globals (declared extern in context.h) ------------------------------
RadarContext      g_ctx;
SemaphoreHandle_t g_state_mux = nullptr;
SemaphoreHandle_t g_tx_mux    = nullptr;

// ---- FreeRTOS tasks ------------------------------------------------------
// All pinned to core 1, drive-base convention. No Bluetooth on this board, but
// keeping the radio core clear costs nothing and keeps the two firmwares'
// task maps reading the same.

static void serialTask(void*) {
  for (;;) {
    proto_poll();                         // drain CDC RX, frame + dispatch lines
    vTaskDelay(pdMS_TO_TICKS(10));        // hello/ping only — 100 Hz is generous
  }
}

static void sensorTask(void*) {
  for (;;) {
    radar_uart_pump();                    // 3 UARTs -> parsers -> g_ctx
    vTaskDelay(pdMS_TO_TICKS(5));         // ~300 B/s per sensor; 5 ms never backlogs
  }
}

static void telemetryTask(void*) {
  TickType_t last = xTaskGetTickCount();
  for (;;) {
    emit_telemetry();                     // snapshot -> rotate -> fuse -> one line
    vTaskDelayUntil(&last, pdMS_TO_TICKS(RADAR_TELEM_PERIOD_MS));
  }
}

void setup() {
  // Native USB CDC. TX buffer must hold a full telemetry frame + a queued log
  // so emitters never block mid-line while holding g_tx_mux (drive-base rule);
  // tx-timeout 0 makes writes DROP when no host is draining the port — the
  // board must keep running headless (boot-before-Mac, Mac app closed).
  Serial.setTxBufferSize(2048);
  Serial.setTxTimeoutMs(0);
  Serial.begin(115200);                   // baud is nominal on native CDC
  delay(50);

  g_state_mux = xSemaphoreCreateRecursiveMutex();
  g_tx_mux    = xSemaphoreCreateMutex();
  g_ctx.boot_id = (uint32_t)esp_random();

  proto_init();
  radar_uart_init();                      // ports + boot config (logs per sensor)

  emit_event_boot(g_ctx.boot_id);         // announce reset (carries boot_id + fw)

  xTaskCreatePinnedToCore(serialTask,    "serial", 4096, nullptr, 3, nullptr, 1);
  xTaskCreatePinnedToCore(sensorTask,    "sensor", 4096, nullptr, 2, nullptr, 1);
  xTaskCreatePinnedToCore(telemetryTask, "telem",  4096, nullptr, 2, nullptr, 1);
}

void loop() {
  // All work lives in the tasks above.
  vTaskDelay(pdMS_TO_TICKS(1000));
}
