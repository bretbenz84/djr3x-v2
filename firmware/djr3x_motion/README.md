# DJ-R3X motion controller — ESP32 firmware (Phase 0)

Real-time motion controller for DJ-R3X's drive base. This is the **Phase 0
bring-up** build: it implements the full Mac↔ESP32 wire protocol
([docs/motion_protocol.md](../../docs/motion_protocol.md) v1) against a **stubbed
hardware layer**, so it runs and is fully testable on a bare ESP32 with **nothing
wired** — no motors, encoders, or ToF sensors required yet.

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
arduino-cli core install esp32:esp32      # Espressif core (3.3.10)
arduino-cli lib install ArduinoJson       # JSON (7.4.3)
```

## Build / upload / monitor

```bash
# Compile (from repo root)
arduino-cli compile --fqbn esp32:esp32:esp32 firmware/djr3x_motion

# Flash the connected board
arduino-cli upload  --fqbn esp32:esp32:esp32 -p /dev/cu.usbserial-10 firmware/djr3x_motion

# Watch the raw NDJSON telemetry stream (Ctrl-A k to quit `screen`)
arduino-cli monitor -p /dev/cu.usbserial-10 -c baudrate=115200
```

## Protocol smoke test (the bring-up acceptance test)

With the board flashed and connected, run the host-side test. It opens the serial
port, drives the protocol, and prints PASS/FAIL for the handshake, telemetry
schema, `turn`/`move` completion, the drive deadman, the heartbeat watchdog +
recovery, estop/clear precedence, error handling, and clamping:

```bash
venv/bin/python firmware/tools/motion_serial_smoketest.py --port /dev/cu.usbserial-10
```

Exit code 0 = every check passed. This is the evidence that the firmware speaks
the contract correctly.

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
