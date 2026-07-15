// tof_matrix_test.ino — bench bring-up firmware for the DFRobot 8x8 Matrix ToF
// (VL53L7CX + onboard RP2040, DFRobot SEN0628) on a SPARE ESP32-WROOM-32.
//
// ⚠ NOT robot firmware. This is a throwaway hardware smoke test: it reads the
//   64-zone depth grid from the sensor over I2C and streams it to the Mac over
//   USB serial, where tools/tof_matrix_monitor.py renders it live. Nothing here
//   is wired into djr3x_motion — the drive base still uses its own VL53L0X/L1X
//   ToF stack (firmware/djr3x_motion/tof.cpp).
//
// The RP2040 on the sensor board does all the ranging math; the ESP32 is just a
// dumb reader that pulls the processed 8x8 grid and forwards it.
//
// ---- Wiring (I2C), per the sensor bench-test notes -------------------------
//   Sensor VCC -> ESP32 3V3
//   Sensor GND -> ESP32 GND
//   Sensor C/R -> ESP32 GPIO22 (SCL)
//   Sensor D/T -> ESP32 GPIO21 (SDA)
//   Add 4.7k pull-ups to 3V3 on SDA/SCL if the board lacks them onboard.
// DIP-set I2C address: 0x33 (the address the sensor is currently strapped to).
//
// ---- Library ---------------------------------------------------------------
//   DFRobot_MatrixLidar — github.com/DFRobot/DFRobot_MatrixLidar
//   (not in the Arduino Library Manager registry; install from the git URL).
//   Confirmed against the driver source: begin()/setRangingMode()/getAllData()
//   all return 0 on success (non-zero on failure); getAllData() writes 64
//   little-endian uint16_t millimetre values, row-major. setRangingMode() has
//   an internal ~5 s settle delay, so first-frame latency after boot is ~5 s.
//
// ---- Wire protocol (USB serial @ 115200, newline-terminated ASCII) ---------
//   "# ..."             human-readable status/diagnostic; the parser ignores it
//   "D,v0,v1,...,v63"   one frame: 64 distances in mm, row-major. The value for
//                       grid cell (row y in 0..7, col x in 0..7) is at index
//                       y*8 + x. 0 typically means "no valid return" in a zone.
//
// ---- Build / flash (ESP32-WROOM-32, CP2102 bridge) -------------------------
//   arduino-cli lib install --git-url https://github.com/DFRobot/DFRobot_MatrixLidar.git
//   arduino-cli compile --fqbn esp32:esp32:esp32 firmware/tof_matrix_test
//   arduino-cli upload  --fqbn esp32:esp32:esp32:UploadSpeed=115200 \
//       -p /dev/cu.usbserial-0001 firmware/tof_matrix_test
//   ./venv/bin/python tools/tof_matrix_monitor.py

#include <Wire.h>
#include <DFRobot_MatrixLidar.h>

static const uint8_t  PIN_SDA           = 21;      // D/T
static const uint8_t  PIN_SCL           = 22;      // C/R
static const uint8_t  TOF_ADDR          = 0x33;    // DIP-strapped address
static const uint32_t FRAME_INTERVAL_MS = 100;     // ~10 Hz stream

DFRobot_MatrixLidar_I2C tof(TOF_ADDR, &Wire);
static uint16_t frame[64];                          // row-major, mm

void setup() {
  Serial.begin(115200);
  delay(300);
  Serial.println();
  Serial.println("# tof_matrix_test — DFRobot 8x8 Matrix ToF (VL53L7CX) bench firmware");
  Serial.printf("# I2C SDA=%u SCL=%u addr=0x%02X\n", PIN_SDA, PIN_SCL, TOF_ADDR);

  Wire.begin(PIN_SDA, PIN_SCL);
  Wire.setClock(400000);

  // Retry forever so a late/flaky bus recovers without needing a re-flash; every
  // retry is visible in the monitor, so a wiring/address/pull-up fault is obvious
  // instead of silent. (begin() returns 0 == OK.)
  for (uint8_t rc; (rc = tof.begin()) != 0; ) {
    Serial.printf("# begin() failed rc=%u — check VCC/GND, SDA=21/SCL=22, "
                  "4.7k pull-ups, and DIP addr=0x%02X. retrying in 1s...\n", rc, TOF_ADDR);
    delay(1000);
  }
  Serial.println("# sensor init OK");

  // setRangingMode() blocks ~5 s internally while the RP2040 reconfigures the
  // VL53L7CX — say so, or the ~5 s of silence looks like a hang.
  Serial.println("# configuring 8x8 ranging mode (~5s settle)...");
  for (uint8_t rc; (rc = tof.setRangingMode(eMatrix_8X8)) != 0; ) {
    Serial.printf("# setRangingMode(8x8) failed rc=%u. retrying in 1s...\n", rc);
    delay(1000);
  }
  Serial.println("# ranging mode = 8x8 — streaming frames (line prefix 'D,')");
}

void loop() {
  static uint32_t last = 0;
  const uint32_t now = millis();
  if (now - last < FRAME_INTERVAL_MS) return;
  last = now;

  const uint8_t rc = tof.getAllData(frame);   // fills 64 uint16_t mm, row-major
  if (rc != 0) {
    Serial.printf("# getAllData() failed rc=%u\n", rc);
    return;
  }

  // Assemble the whole "D,..." line in one buffer and write it in a single call
  // so it can't interleave with anything else on the wire. Worst case is
  // 1 + 64*"65535,"-ish = ~385 chars; 400 is comfortably clear.
  char line[400];
  int n = 0;
  line[n++] = 'D';
  for (int i = 0; i < 64; i++) {
    n += snprintf(line + n, sizeof(line) - n, ",%u", frame[i]);
  }
  line[n++] = '\n';
  Serial.write(reinterpret_cast<const uint8_t*>(line), n);
}
