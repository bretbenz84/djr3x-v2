# djr3x_radar — LD2450 bearing-prior ring (ESP32-S3)

Firmware for the 360° mmWave radar ring on DJ-R3X's base: up to
`RADAR_SENSOR_COUNT` HLK-LD2450 modules (one hardware UART each), fused
on-board into a robot-frame `(bearing, range, confidence)` target list and
streamed to the Mac at 10 Hz over **native USB CDC**. Feature spec:
[docs/radar-bearing-prior-spec.md](../../docs/radar-bearing-prior-spec.md).
The Mac consumer is `hardware/radar.py`.

The radar is a **bearing prior, not a detector** — it tells the come-here
search where to look first; the camera confirms. This board moves nothing, so
there is no watchdog/estop machinery: it boots streaming (no handshake
required) and reports `radar.ok:false` honestly when no sensor is delivering.

## Board

ESP32-S3 dev board, **N16R8** module (16 MB flash, 8 MB *octal* PSRAM).
Octal PSRAM bonds **GPIO 33–37 to the in-package dies — never use them**
(`pins.h` avoids them along with strapping/USB/flash pins). The Mac link is
the **native USB port** (the S3's USB-OTG/JTAG connector), not the "COM" /
UART bridge port — that's what frees all three hardware UARTs for sensors.

## Build modes

Like the drive base, the repo default builds against a **stub** so a bare S3
(modules not wired) compiles, runs, and streams — the stub synthesizes a
scripted scene (one person orbiting through every seam + a second stepping in
at −90°), encodes it into real LD2450 wire bytes, and feeds them through the
same parser/fusion pipeline. Flags are passed per-build, never by editing
source:

```bash
# Stub (bare board — repo default):
arduino-cli compile --fqbn esp32:esp32:esp32s3:CDCOnBoot=cdc firmware/djr3x_radar

# Real sensors wired:
arduino-cli compile --fqbn esp32:esp32:esp32s3:CDCOnBoot=cdc \
  --build-property "compiler.cpp.extra_flags=-DRADAR_HW_PRESENT=1" \
  firmware/djr3x_radar
```

`CDCOnBoot=cdc` is **required** — without it `Serial` maps to UART0 and the
native USB port stays silent. Flash + monitor (always `compile --upload`, never
a bare `upload` — `upload` can't take `--build-property` and would flash the
last cached variant):

```bash
arduino-cli compile --upload --fqbn esp32:esp32:esp32s3:CDCOnBoot=cdc \
  --build-property "compiler.cpp.extra_flags=-DRADAR_HW_PRESENT=1" \
  -p "$PORT" firmware/djr3x_radar
arduino-cli monitor -p "$PORT" -c baudrate=115200
```

`$PORT` is the CDC device (`/dev/cu.usbmodem*`). The Mac runtime matches this
board by **USB serial number** (`RADAR_ESP32_SERIAL` in `.env`), not by path —
find it with `venv/bin/python -c "import serial.tools.list_ports as lp;
[print(p.device, p.serial_number) for p in lp.comports()]"`.

> **If the port vanishes:** a firmware crash takes the native USB CDC device
> down entirely — the Mac sees the port *disappear*, not error. Hold the BOOT
> button while plugging the board in to re-enter the ROM bootloader, flash a
> good build, and the port returns. (Drive-base boards don't do this; their
> CP2102 bridge stays enumerated through any crash.)

## Wiring

`pins.h` is the one wiring file — a table of
`{uart, tx_pin, rx_pin, mount_angle_deg}` rows, written for N sensors (run 2
during bring-up by dropping a row and `RADAR_SENSOR_COUNT`). Defaults:

| Sensor | Mount (robot frame) | UART | ESP RX ← module TX | ESP TX → module RX |
| --- | --- | --- | --- | --- |
| S0 | 0° (front) | 1 | GPIO 4 | GPIO 5 |
| S1 | +120° (left-rear) | 2 | GPIO 6 | GPIO 7 |
| S2 | −120° (right-rear) | 0 | GPIO 8 | GPIO 9 |

Mount angles use the project-wide sign convention
([docs/motion_protocol.md](../../docs/motion_protocol.md) §4): 0 = robot
forward, **+ = left/CCW**, wrapped (−180, 180]. Both TX and RX are wired per
sensor so config can be pushed without pulling a module out of the ring.
Sensors run 256000 8N1, 5 V supply / 3.3 V TTL — powered from USB 5 V through
the S3 board (~80 mA each), **not** the robot's 5 V rail.

At boot (real build, `RADAR_SENSOR_BOOT_CONFIG`), each sensor gets one
config-mode transaction: read the module firmware version into the logs, force
**multi-target tracking** (a module left in single-target mode would silently
cap the ring at one person), and turn the module's **Bluetooth off**
(`RADAR_DISABLE_BLUETOOTH`) — it ships on for the HLKRadarTool phone app, which
this build never uses.

The Bluetooth bit is the one **persistent** thing boot config writes, and per
the protocol doc it only goes live **on the module's next restart**. We don't
send a reboot to force it: the modules run off the S3's 5 V, so they restart
with the board, and every boot after the next power-cycle starts with the radio
already dark. The command is re-asserted every boot, so a module swapped into
the ring gets configured without anyone remembering to.

Consequence worth knowing before you flash: once this has taken effect, **the
HLKRadarTool phone app can no longer reach those modules** — the only way back
is `RADAR_DISABLE_BLUETOOTH 0` plus a reflash. The boot log reports it per
sensor as `bt=off@next-boot`, `bt=NO-ACK`, or `bt=skipped`.

## Protocol

Same NDJSON contract as the drive base (`v:1`, `type`/`cmd`, unknown fields
ignored). Board→Mac: `hello` (caps `["radar"]` — deliberately **not** `drive`,
so `setup_macos.sh`'s motion-base probe can't misidentify this board), 10 Hz
`telemetry`, `event:boot`, `log`. Mac→board: `hello`, `ping` (accepted,
ignored), anything else acks `unknown_cmd`.

```json
{"v":1,"type":"telemetry","t":12834,
 "radar":{"ok":true,"up":3,"targets":[
   {"b":137.2,"r":4.10,"c":0.82,"s":-0.30,"m":6}]},
 "sens":[{"ok":true,"frames":1201,"bad":0,"drop":0}, ...],
 "errs":0}
```

Per target: `b` bearing (deg, + = left/CCW), `r` range (m), `c` confidence
(0–1, falls toward each sensor's ±60° FOV edges, raised when two sensors agree
across a seam), `s` radial speed (m/s, + = away — unofficial, verify), `m`
bitmask of contributing sensors. Targets are sorted best-first. `radar.ok`
false = **no sensor is delivering** ("I can't see"), distinct from an empty
`targets` list ("I see nobody").

## Parser provenance (do not "fix" from another driver)

The LD2450 byte layout in `ld2450.h`/`ld2450.cpp` was cross-checked against
Hi-Link's official protocol doc V1.03 (worked examples), the Hi-Link operation
manual, and the ESPHome core `ld2450` component. The sign encoding is
sign-and-magnitude with an **inverted flag** (high bit 1 = positive); two
popular open-source drivers (csRon/HLK-LD2450, TillFleisch/ESPHome-HLK-LD2450)
decode x/speed with the **opposite polarity** to the official doc's own worked
example. Which side of the sensor is +x is stated nowhere official — if
bring-up shows mirrored bearings, set `RADAR_FLIP_X` in `calib.h`, don't touch
the math. Full source notes at the top of `ld2450.h`.

**Settled 2026-08-14 on hardware:** bearings came out mirrored left/right, so
this batch of modules uses the drivers' polarity, not the official doc's.
`RADAR_FLIP_X` is now **1**. The parser math still implements the official
convention — the flag is the only thing that moved, which is what keeps the
provenance above honest. A future batch that reads correct again is a one-line
revert, not a rewrite.

## Testing without hardware

- `tests/test_radar_parser.py` compiles `ld2450.cpp` + `fusion.cpp` (both
  Arduino-free by contract) with clang++ into a host binary
  (`firmware/tools/radar_parse_host.cpp`) and drives the spec's scenarios —
  seam crossing, two targets, zero targets, malformed frame, mid-frame
  truncation — through the same C++ the S3 runs. This is the parser's
  regression harness; keep it green.
- `firmware/tools/ld2450_synth.py` generates official-encoding frames for
  those tests and for anything that wants scripted sensor bytes.
- The stub build runs the full pipeline on a bare S3 over real USB.
- `firmware/tools/radar_serial_smoketest.py` checks a flashed board
  end-to-end (handshake, telemetry schema, target sanity, error handling).
- The GUI's Motivator Control window (`main.py --gui` → 🕹 MOTIVATOR) has a
  live **RADAR RING** scope — targets, FOV wedges, per-sensor health — fed
  from `hardware/radar.py` (docs/GUI.md). The fastest way to eyeball a flashed
  board: the stub build's scripted scene animates the scope with no modules
  wired.
