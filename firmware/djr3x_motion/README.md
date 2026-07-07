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
  integrated from encoder deltas. **ToF defaults to a safe "clear" stub** here, so
  **obstacle avoidance is inactive** — add `-DMOTION_TOF_PRESENT=1` once the 8 ToF
  sensors (4× VL53L0X + 4× VL53L1X) are wired (driver in `tof.cpp`; TCA9548A mux
  addressing — see "Build modes"). Pins live in `pins.h`, measured/tuned constants in
  `calib.h` (the geometry values there are **placeholders — measure them on the real
  base**, docs §14).

- The plant model in `control.cpp` synthesizes odometry from commanded velocity,
  so `turn`/`move`/`come` actually run to completion and emit `done`.
- `hal.cpp` returns a "clear room" for the ToF sensors, so the reflex/zone logic
  stays green.
- Everything above the HAL (protocol, control, safety, state machine, watchdog,
  deadman) is the real thing and is what we validate here.

As hardware is wired, flip `MOTION_HW_PRESENT` to `1` in [`hal.h`](hal.h) and fill
the marked driver sections — nothing above the HAL changes.

## Board

Classic ESP32-WROOM-32 (Elegoo DevKit, CP2102 USB bridge). FQBN `esp32:esp32:esp32`.
The serial-port path varies per machine — find it with `arduino-cli board list` (it is
the same path you set as `MOTION_ESP32_PORT` in `.env`). The commands below use `$PORT`:

```bash
export PORT=/dev/cu.usbserial-XXXX   # YOUR board's port — see `arduino-cli board list`
```

## Toolchain (already installed on this machine)

```bash
arduino-cli core install esp32:esp32      # Espressif core (3.3.10 on this machine)
arduino-cli lib install ArduinoJson       # JSON (7.4.3)
arduino-cli lib install ESP32Encoder      # Hall quadrature decode (0.12.0) — live build only
arduino-cli lib install VL53L0X           # Pololu short-range ToF (1.3.1) — ToF build (-DMOTION_TOF_PRESENT=1)
arduino-cli lib install VL53L1X           # Pololu long-range ToF (1.3.1)  — ToF build (-DMOTION_TOF_PRESENT=1)
```

> **Core version (2.x and 3.x both supported).** `setup_macos.sh` installs
> `esp32:esp32` **unpinned**, so a machine gets whatever core is current — older installs
> may be on the legacy **2.x** core, newer ones on **3.x** (e.g. 3.3.10). The live HAL
> builds on **both**: the LEDC API moved from channel-based (2.x: `ledcSetup` /
> `ledcAttachPin`) to pin-based (3.x: `ledcAttach`), and `hal.cpp` picks the right one
> via an `ESP_ARDUINO_VERSION` guard. `ESP32Encoder` 0.12.0 covers the matching PCNT
> change. The `setup_macos.sh` toolchain step installs all three deps automatically.

## Build / upload / monitor

```bash
# Compile (from repo root)
arduino-cli compile --fqbn esp32:esp32:esp32 firmware/djr3x_motion

# Flash the connected board
arduino-cli upload  --fqbn esp32:esp32:esp32 -p "$PORT" firmware/djr3x_motion

# Watch the raw NDJSON telemetry stream (Ctrl-A k to quit `screen`)
arduino-cli monitor -p "$PORT" -c baudrate=115200
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
  -p "$PORT" firmware/djr3x_motion
```

The live build boots to **idle with the motors disabled** — nothing moves until an
explicit command energizes them.

### ToF obstacle avoidance (sensors wired)

Off by default even in the live build (`hal_read_tof` reports a clear room). Enable it
once the 8 ToF sensors are on the I²C mux — combine the flags. The layout (rev 2) is
**8 radial sensors** at the 540 mm base-ring surface, every 45° starting 22.5° off the
forward axis: 4 short-range **VL53L0X** on mux ch 0-3 as the LEFT/RIGHT pairs
(`lf,lb,rf,rb` — lateral clearance for the hallway steering assist) + 4 long-range
**VL53L1X** on mux ch 4-7 as the FRONT/REAR pairs (`fl,fr,rl,rr`, ±22.5° off each axis —
stop reflex + room sense). All stay at 0x29; the TCA9548A mux selects one channel at a
time (zero XSHUT GPIOs). There is **no down/cliff sensor** in this layout, so cliff/
drop-off detection is unavailable. While driving FORWARD on the gamepad, the firmware
auto-steers away from walls / centers in a hallway (`assist_*` config params, docs §6.4);
the operator's stick adds on top and B/e-stop + the stop reflex always win.

```bash
arduino-cli compile --fqbn esp32:esp32:esp32 \
  --build-property "compiler.cpp.extra_flags=-DMOTION_HW_PRESENT=1 -DMOTION_TOF_PRESENT=1" \
  firmware/djr3x_motion
```

The mux is **required** for this layout (8 sensors exceed the ESP32's free XSHUT GPIOs;
the `-DMOTION_TOF_USE_MUX=0` path `#error`s). `tof.cpp` is still a **scaffold** — not yet
fully hardware-validated; bench-check the channel→field order and timing budgets (docs §6).
Bring-up emits one `[motion_fw] tof[…]` log per sensor (OK/FAIL) + an `N/8 up` tally.

### Manual gamepad override (Bluepad32) — Phase 1.5

A Bluetooth gamepad paired **directly to the ESP32** (not the Mac) can grab the wheel
and override autonomous/voice motion — it works even with the USB link down (docs §11).
Off by default; the feature is `gamepad.cpp` behind `MOTION_GAMEPAD_PRESENT`.

**This build needs a different toolchain.** Bluepad32 replaces the Bluetooth stack, so
it ships as its own ESP32 board package, *not* a library on `esp32:esp32`. Install it
once, then build with that FQBN:

```bash
arduino-cli config add board_manager.additional_urls \
  https://raw.githubusercontent.com/ricardoquesada/esp32-arduino-lib-builder/master/bluepad32_files/package_esp32_bluepad32_index.json
arduino-cli core update-index
arduino-cli core install esp32-bluepad32:esp32   # platform ID is HYPHENATED (FQBN below too)

# Use `compile --upload` (NOT a bare `upload` — that has no --build-property, so it would
# flash whatever variant was last cached). UploadSpeed=115200 is REQUIRED: this USB bridge
# fails the default 921600 ("Invalid head of packet" right after the baud change).

# CONNECTIVITY TEST — gamepad ON, motors STUBBED (the base will NOT move; safest for a
# first pairing test). The live `gp` telemetry + GUI mirror run identically to live:
arduino-cli compile --fqbn esp32-bluepad32:esp32:esp32:UploadSpeed=115200 \
  --build-property "compiler.cpp.extra_flags=-DMOTION_GAMEPAD_PRESENT=1" \
  --upload -p "$PORT" firmware/djr3x_motion

# LIVE drive base + gamepad (only on a stand; add -DMOTION_TOF_PRESENT=1 if ToF is wired):
arduino-cli compile --fqbn esp32-bluepad32:esp32:esp32:UploadSpeed=115200 \
  --build-property "compiler.cpp.extra_flags=-DMOTION_HW_PRESENT=1 -DMOTION_GAMEPAD_PRESENT=1" \
  --upload -p "$PORT" firmware/djr3x_motion
```

The dual-version LEDC guard in `hal.cpp` means the motor code builds on the Bluepad32
core whether it tracks ESP32 2.x or 3.x.

**8BitDo Pro 2 — pairing.** Power it in a Bluepad32-friendly mode (hold **START + A**
for the Android/"D" profile; if pairing is flaky try **START + X** for X-input), then
hold the top pair button until the LEDs sweep. Bluepad32 accepts the new connection and
remembers the bond for next time.

**Controls** (`gamepad.cpp`, tunables in `calib.h`):

| Input | Action |
| --- | --- |
| Left stick | arcade drive — Y forward/back, X turn. Pure X (no Y) spins in place at full turn authority; as Y is added the spin **blends smoothly** into an arcade arc (turn authority eases to the speed level, inside-wheel reverse eases out — no hard regime snap) |
| **L3 (click left stick)** | **cycle drive speed level**: slow (default) → faster → full. Latches; resets to slow on reconnect. Emits `event:"speed" level:1..3` |
| **D-pad** | **spin to an absolute heading** (encoder test): Up=0°, Left=+90° (CCW), Down=180°, Right=−90° (CW) |
| **B** | **E-STOP** (always honored) |
| Start | clear e-stop + return control to AUTO |
| Hold **both** triggers (L2+R2) | FULL-OVERRIDE: bypass ToF gating (nudge through tight spots) |

**D-pad → absolute-heading turn (encoder validation).** Each arrow press spins the base in
place to the absolute heading above, via the same encoder-closed-loop finite `turn` the Mac
uses — so on a correctly wired + calibrated base it lands square at 90° steps. Use it to
sanity-check the encoders: a **flipped encoder sign** makes the spin run away (never
converges) instead of stopping; a wrong `counts_per_meter` / `track_width_m` scale makes it
over- or under-rotate (e.g. command 90°, get 75°). It is issued as a **MANUAL** turn
(`ctl_manual_turn`), so it runs locally on the base — it survives a USB drop and the Mac
won't fight it — and it's computed as a shortest-path delta from the live heading
(`g_ctx.odom.theta`). A left-stick push cancels an in-flight turn; **B** e-stops it; **Start**
clears + returns to AUTO. A pure spin has no linear travel, so **ToF does NOT gate it** — run
on a clear floor or a stand. (Needs the `-DMOTION_HW_PRESENT=1` build for real encoders; on
the stub build the turn "works" but only against synthesized odometry, testing nothing.)

Any meaningful stick push switches `owner` to **MANUAL** — the Mac's drive/turn/move/come
are then refused (`stop`/`estop`/`config`/`ping` still work) and the GUI shows
`owner: manual`. Default is **MANUAL-ASSISTED** (ToF still protects you); FULL-OVERRIDE
is the only way past it and only while held. If the pad drops, the base stops immediately
and stays MANUAL until `MOTION_MANUAL_AUTORETURN`'s idle timeout hands back to AUTO (or
you reconnect and press Start).

**Action buttons → R3X soundboard / animations.** The buttons motion does NOT use —
**A, X, Y, Select (−), Home (★), R3 (right stick click)** — are forwarded to the
Mac as `event:"button"` (rising edge, one per press) by `poll_action_buttons()`.
(L3 is NOT forwarded — it's the speed-level toggle above.) They fire
**whenever the pad is connected**, independent of the drive `owner` (so the soundboard works
in AUTO and pressing them does NOT grab the wheel). On the Mac,
`intelligence/motion_controller._on_motion_event` looks the button up in
`config.MOTION_GAMEPAD_BUTTON_ACTIONS` and triggers a **sound clip** (`audio/soundboard.py`
plays an MP3 from `assets/audio/clips/`) and/or a **servo animation**
(`sequences.animations.play_body_beat`). The map is data-driven — edit it to remap, no
firmware change. Button names: `a x y select home l3 r3`. (The **D-pad is NOT** in this list
— it drives the encoder-test heading turns above, not the soundboard.) (Needs the
`-DMOTION_GAMEPAD_PRESENT=1` build above.)

**Live state in the GUI ("Motivator Control").** Every telemetry frame carries a `gp`
object — `{connected, lx, ly, btn}` — mirroring the pad's left stick (normalized, right=+
/ stick-up=+) and a pressed-button bitmask (`emit_telemetry` in `proto_io.cpp`, captured
each tick in `gamepad_tick`). The GUI's **PHYSICAL CONTROLLER** panel
(`GamepadMirrorWidget` in `gui/dashboard.py`) renders it read-only: a dot tracks the stick
and held buttons light up, refreshed at the dialog's 150 ms telemetry tick. The bitmask
bit order (`GP_BTN_*` in `gamepad.cpp`) MUST stay in sync with `_GP_BTN_LABELS` in the GUI:
`A B X Y L1 R1 L2 R2 ↑ ↓ ← → Sel Start Home L3 R3`. When no pad is paired (or in a
non-gamepad build) `gp.connected` is false and the panel shows "no pad connected". This is
the recommended way to verify controller connectivity — pair, open Motivator Control, and
watch the dot move.

> **Scaffold:** the arbitration (owner switching, full-override, disconnect failsafe,
> watchdog-bypass-while-manual) is compiled and verified in the stock build, but the
> Bluepad32 I/O itself is **not yet hardware-validated** — verify the button map and
> pairing on the real pad. Builds only with the `esp32-bluepad32` core above.

## Protocol smoke test (the bring-up acceptance test)

With the board flashed and connected, run the host-side test. It opens the serial
port, drives the protocol, and prints PASS/FAIL for the handshake, telemetry
schema, `turn`/`move` completion, the drive deadman, the heartbeat watchdog +
recovery, estop/clear precedence, error handling, and clamping:

```bash
venv/bin/python firmware/tools/motion_serial_smoketest.py --port "$PORT"
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
| `hal.{h,cpp}` | Hardware abstraction — motor/encoder stubs vs real drivers behind `MOTION_HW_PRESENT` |
| `tof.cpp` | ToF (VL53L0X ×5) driver — clear-room stub vs XSHUT/mux real driver behind `MOTION_TOF_PRESENT` |
| `gamepad.cpp` | Bluetooth gamepad manual override (Bluepad32) behind `MOTION_GAMEPAD_PRESENT` |

## Concurrency model

Four FreeRTOS tasks share `g_ctx` behind a recursive mutex (`g_state_mux`); all
serial writes are serialized behind `g_tx_mux` so NDJSON lines never interleave.
Lock-order rule: **never hold `g_state_mux` while emitting** (emit takes
`g_tx_mux`) — snapshot under the state lock, release, then emit.

## Phase roadmap

See [docs/motion_system.md](../../docs/motion_system.md) §17. This is Phase 0;
Phase 1 wires motors + encoders + ToF and replaces the HAL stubs with real
drivers + per-wheel PID.

## Battery sense (INA226, optional)

Pack voltage (12.8V 4S LiFePO4) is read by an INA226 breakout on the existing
I2C bus — no voltage divider needed (the INA measures up to 36V at its VBUS pin,
covering the charger's 14.6V peak). Without the sensor, telemetry reports
`batt_mv: -1` and the Mac-side battery feature stays dormant.

Wiring (voltage-only, four wires):

    INA226 VCC  -> 3V3
    INA226 GND  -> GND (common with the pack)
    INA226 SDA  -> GPIO 21   (piggyback the ToF bus; INA addr 0x40, mux is 0x70)
    INA226 SCL  -> GPIO 22
    INA226 VBUS -> BATT+ (the pack's positive terminal)

Current sensing (optional, later): the stock module shunt (R100 = 100 mΩ) only
ranges ±0.8 A — useless for drive motors. Fit a 2 mΩ shunt inline in the main
battery lead, then build with `-DBATT_SHUNT_MICROOHM=2000`; telemetry gains a
real `batt_ma` and the Mac side can coulomb-count true state of charge (LiFePO4
voltage is too flat mid-pack for voltage-only percentages).
