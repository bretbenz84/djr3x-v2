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
#include "battery.h"
#include "imu.h"
#include "env.h"
#include "mag.h"
#include "i2c_trunk.h"

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
  uint8_t batt_div = 0;
  uint8_t env_div = 0;
  uint8_t mag_div = 0;
  for (;;) {
    TofMm t;
    hal_read_tof(t);                      // stub: all clear
    LOCK_STATE(); g_ctx.tof = t; UNLOCK_STATE();
    imu_tick(0.02f);                      // LSM6DS3 attitude @ 50 Hz (no-op if absent)
    if (++batt_div >= 50) {               // 1 Hz is plenty for a 20Ah pack
      batt_div = 0;
      battery_tick();
    }
    if (++env_div >= 100) {               // room climate: 0.5 Hz is generous
      env_div = 0;
      env_tick();
    }
    if (++mag_div >= 5) {                 // compass raw axes: 10 Hz (host fuses)
      mag_div = 0;
      mag_tick();
    }
    vTaskDelayUntil(&last, pdMS_TO_TICKS(20));   // 50 Hz
  }
}

static void telemetryTask(void*) {
  TickType_t last = xTaskGetTickCount();
  for (;;) {
    emit_telemetry();
    // 10 Hz (was 20): a ~480 B frame at 20 Hz was ~84% of the 115200-baud line —
    // no headroom, so pad-driving load backed frames up and the GUI showed stale
    // data. Every consumer is a latest-snapshot reader (GUI ticks at ~6.7 Hz),
    // so 10 Hz keeps everyone fresher than they can consume at ~42% line util.
    vTaskDelayUntil(&last, pdMS_TO_TICKS(100));
  }
}

void setup() {
  // TX buffer must hold a full telemetry frame (+ a queued ack/log) so emitters
  // never block mid-line while holding g_tx_mux — at 115200 baud a queued frame
  // takes ~35 ms to drain, and a blocked mux stalls every other emitter with it.
  // MUST be called before begin().
  Serial.setTxBufferSize(2048);
  Serial.begin(115200);
  delay(50);

  g_state_mux = xSemaphoreCreateRecursiveMutex();
  g_tx_mux    = xSemaphoreCreateMutex();

  g_ctx.boot_id     = (uint32_t)esp_random();
  g_ctx.last_mac_ms = millis();           // grace before the watchdog can arm
  g_ctx.seen_mac    = false;              // watchdog stays disarmed until first Mac line

  hal_init();
  i2c_trunk_init();                       // single owner of shared GPIO21/22 Wire bus
  hal_tof_init();
  battery_init();                         // INA226 probe (shares the ToF I2C bus)
  imu_init();                             // LSM6DS3 probe + gyro bias cal (same bus)
  env_init();                             // BMP280/BME280 probe (same bus)
  mag_init();                             // QMC5883L probe (same bus)
  proto_init();
  control_init();
  safety_init();
  gamepad_init();

  emit_event_boot(g_ctx.boot_id);         // announce reset (carries boot_id + fw)

  // Highest priority = control; serial just under it. ALL our tasks live on core 1:
  // the Bluetooth controller + BTstack (Bluepad32 builds) own core 0 at high priority,
  // and a CONNECTED gamepad streams input reports continuously — with sensor/telemetry
  // pinned to core 0 they starved whenever a pad was connected (field bug 2026-07-12:
  // battery menu bar and GUI sensors went stale the moment the pad linked). Core 1 has
  // the headroom: control ~5%, serial poll ~5%, sensor I2C ~30% worst, telemetry ~2%,
  // leaving the Arduino loopTask (gamepad poll, prio 1) plenty of gaps.
  xTaskCreatePinnedToCore(controlTask,   "control", 4096, nullptr, 4, nullptr, 1);
  xTaskCreatePinnedToCore(serialTask,    "serial",  4096, nullptr, 3, nullptr, 1);
  xTaskCreatePinnedToCore(sensorTask,    "sensor",  3072, nullptr, 2, nullptr, 1);
  xTaskCreatePinnedToCore(telemetryTask, "telem",   4096, nullptr, 2, nullptr, 1);

  // The Arduino loopTask (which services Bluetooth via BP32.update in loop()) is
  // created at priority 1 — BELOW the sensor task's 2 on the same core. Field
  // regression after the core-1 consolidation: whenever I2C stalled (a flaky
  // IMU disturbing the shared trunk stretched every ToF/INA transaction
  // toward its timeout), the sensor task monopolized the core, BP32.update()
  // starved, and the GAMEPAD DISCONNECTED right as the current monitor failed.
  // Raise ourselves (setup runs ON loopTask) above the sensor/telemetry tier so
  // Bluetooth servicing always gets the CPU it needs; sensor I2C runs in the gaps.
  vTaskPrioritySet(NULL, 3);
}

void loop() {
  // The control/serial/sensor/telemetry work runs in the tasks above. The Arduino
  // loopTask polls the Bluetooth gamepad here (BP32.update needs frequent calls); with
  // the gamepad feature off, gamepad_tick() is a no-op and this idles slowly.
  gamepad_tick();
  vTaskDelay(pdMS_TO_TICKS(GAMEPAD_POLL_MS));
}
