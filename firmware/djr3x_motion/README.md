# DJ-R3X motion controller — ESP32 firmware (Phase 0)

Real-time motion controller for DJ-R3X's drive base. It implements the full
Mac↔ESP32 wire protocol ([docs/motion_protocol.md](../../docs/motion_protocol.md)
v1) and has **two build modes** (see "Build modes" below):

- **Stub (`MOTION_HW_PRESENT=0`, the repo default)** — the original **Phase 0
  bring-up** build: a stubbed hardware layer, so it runs and is fully testable on a
  bare ESP32 with **nothing wired**. The plant model synthesizes odometry; ToF reads
  "clear".
- **Live (`-DMOTION_HW_PRESENT=1`)** — the **Phase 1 real drive base**: BTS7960 PWM,
  Hall quadrature encoders (PCNT/ESP32Encoder), per-wheel velocity PID, and odometry
  integrated from encoder deltas. **ToF is still a safe "clear" stub** (the 5×
  VL53L0X subsystem + its addressing scheme are a later step), so **obstacle
  avoidance is inactive in this build.** Pins live in `pins.h`, measured/tuned
  constants in `calib.h` (the geometry values there are **placeholders — measure
  them on the real base**, docs §14).

- The plant model in `control.cpp` synthesizes odometry from commanded velocity,
  so `turn`/`move`/`come` actually run to completion and emit `done`.
- `hal.cpp` returns a "clear room" for the ToF sensors, so the reflex/zone logic
  stays green.
- Everything above the HAL (protocol, control, safety, state machine, watchdog,
  deadman) is the real thing and is what we validate here.

As hardware is wired, flip `MOTION_HW_PRESENT` to `1` in [`hal.h`](hal.h) and fill
the marked driver sections — nothing above the HAL changes.

## Board

Classic ESP32-WROOM-32 (Elegoo DevKit, CP2102 USB bridge). FQBN `esp32:esp32:esp32`,
typically on `/dev/cu.usbserial-10`.

## Toolchain (already installed on this machine)

```bash
arduino-cli core install esp32:esp32      # Espressif core (2.0.17 on this machine)
arduino-cli lib install ArduinoJson       # JSON (7.4.3)
arduino-cli lib install ESP32Encoder      # Hall quadrature decode (0.12.0) — live build only
```

> **Core version matters.** This machine has Arduino-ESP32 **2.0.17**. The LEDC and
> PCNT APIs differ in core 3.x, so the Phase-1 HAL is written against 2.0.x. The
> `setup_macos.sh` toolchain step installs all three deps automatically.

## Build / upload / monitor

```bash
# Compile (from repo root)
arduino-cli compile --fqbn esp32:esp32:esp32 firmware/djr3x_motion

# Flash the connected board
arduino-cli upload  --fqbn esp32:esp32:esp32 -p /dev/cu.usbserial-10 firmware/djr3x_motion

# Watch the raw NDJSON telemetry stream (Ctrl-A k to quit `screen`)
arduino-cli monitor -p /dev/cu.usbserial-10 -c baudrate=115200
```

## Build modes

`MOTION_HW_PRESENT` is `#ifndef`-guarded in [`hal.h`](hal.h) and defaults to `0`,
so a plain build (and the smoke test, and `tests/`) stays bare-board. Select the
live drivers with a build-property override — no source edit:

```bash
# Stub / bare-board (default)
arduino-cli compile --fqbn esp32:esp32:esp32 firmware/djr3x_motion

# Live Phase-1 drive base
arduino-cli compile --fqbn esp32:esp32:esp32 \
  --build-property "compiler.cpp.extra_flags=-DMOTION_HW_PRESENT=1" firmware/djr3x_motion

# Flash live (115200 — 921600 is unreliable on this USB bridge)
arduino-cli upload --fqbn esp32:esp32:esp32:UploadSpeed=115200 \
  --build-property "compiler.cpp.extra_flags=-DMOTION_HW_PRESENT=1" \
  -p /dev/cu.usbserial-3110 firmware/djr3x_motion
```

The live build boots to **idle with the motors disabled** — nothing moves until an
explicit command energizes them.

## Protocol smoke test (the bring-up acceptance test)

With the board flashed and connected, run the host-side test. It opens the serial
port, drives the protocol, and prints PASS/FAIL for the handshake, telemetry
schema, `turn`/`move` completion, the drive deadman, the heartbeat watchdog +
recovery, estop/clear precedence, error handling, and clamping:

```bash
venv/bin/python firmware/tools/motion_serial_smoketest.py --port /dev/cu.usbserial-3110
```

Exit code 0 = every check passed. This is the evidence that the firmware speaks
the contract correctly.

> **Run the smoke test against the STUB build (`MOTION_HW_PRESENT=0`).** Its
> `turn`/`move` checks expect a `done` reply, which the stub plant produces from
> synthesized odometry. In a **live build on a bare board** those commands never
> complete (no encoder counts → no progress), so the test would hang/fail on them —
> by design. Only run it live once real wheels + encoders are wired and the base is
> on a stand.

## File map

| File | Role |
| --- | --- |
| `djr3x_motion.ino` | Globals, FreeRTOS task setup (control/serial/sensor/telemetry), boot |
| `protocol.h` | Wire version, capabilities, enum↔string maps (the vocabulary) |
| `context.h` | Shared `MotionContext` state, params, hard caps, lock macros |
| `config_params.{h,cpp}` | `config` command application + clamping |
| `proto_io.{h,cpp}` | NDJSON framing, parse + dispatch, all emitters |
| `control.{h,cpp}` | Stub plant + odometry, finite-command lifecycle |
| `safety.{h,cpp}` | Watchdog, ToF zones + reflex stop, comms-loss handling |
| `hal.{h,cpp}` | Hardware abstraction — stubs today, real drivers behind `MOTION_HW_PRESENT` |

## Concurrency model

Four FreeRTOS tasks share `g_ctx` behind a recursive mutex (`g_state_mux`); all
serial writes are serialized behind `g_tx_mux` so NDJSON lines never interleave.
Lock-order rule: **never hold `g_state_mux` while emitting** (emit takes
`g_tx_mux`) — snapshot under the state lock, release, then emit.

## Phase roadmap

See [docs/motion_system.md](../../docs/motion_system.md) §17. This is Phase 0;
Phase 1 wires motors + encoders + ToF and replaces the HAL stubs with real
drivers + per-wheel PID.
