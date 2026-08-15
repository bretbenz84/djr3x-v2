# DJ-R3X: Radar Bearing Prior (LD2450 ring)

## Goal

When R3X gets a "come here" or otherwise needs to locate a person, he currently
turns haphazardly until the camera finds a face. Add a 360° mmWave radar ring to
the base that supplies a **bearing prior** — a coarse "start looking at 137°"
hint — so the head/body turn is directed instead of random.

The radar is **not** the detector. YOLOv11n + dlib remain the detector and
confirm or reject. Radar only decides where to look first. Design for that: a
noisy or occasionally wrong bearing costs one wasted head turn, not a
misidentification.

## Hardware

- 3× HLK-LD2450 24 GHz FMCW radar modules
- 1× ESP32-S3 dev board, already connected to the Mac Mini over USB
- Modules mount inside the base ring 120° apart, behind 3 mm PETG
  (radar-transparent; no cutouts). **Orientation revised 2026-08-15:** two
  forward-quarter modules at ±60° plus one rear module at 180° ("two in front,
  one in back"), replacing the original 0° / 120° / 240° layout (one forward,
  two rear). The seams are therefore at 0° and ±120°. `firmware/djr3x_radar/pins.h`
  is the authoritative table.
- Each module: ±60° azimuth, ±35° tilt, 8 m range, 0.75 m range resolution,
  2°–20° angle accuracy, UART at 256000 baud

**The radar modules have not arrived yet — they land tomorrow.** Everything up
to and including the Mac-side consumer must be buildable and testable today
against synthetic data. See "Phasing" below.

## Before writing any code

1. Read the existing ESP32 drive-base firmware. Match its structure, build
   system (PlatformIO vs Arduino IDE vs ESP-IDF), framing/checksum conventions,
   and logging style. Do not introduce a second firmware idiom.
2. Read the Mac-side serial transport that talks to the drive base. The radar
   board should reuse that layer, not get a parallel one.
3. Read the spatial awareness / `places.db` module and the existing head-turn
   and person-search behavior. Find where the current haphazard scan is
   initiated — that is the single integration point.
4. Read how ToF sensor data is currently consumed.

Report what you found and your integration plan **before** implementing.

## Protocol — do not guess

The LD2450 emits fixed-length binary frames (header, three 8-byte target slots
carrying x/y/speed/distance-resolution, tail) at roughly 10 Hz. Empty target
slots are zero-filled — "up to 3 targets" comes from filtering, not from
variable frame length.

I have not verified the exact byte layout, endianness, or sign encoding. **Do
not implement the parser from memory or from an Amazon listing.** Locate
Hi-Link's LD2450 serial protocol document or a well-maintained open-source
driver, cite what you used, and note any point where sources disagree. Flag it
for me rather than guessing.

## Deliverables

### 1. ESP32-S3 firmware

- Three hardware UARTs, one per sensor. Native USB CDC handles the Mac link, so
  all three UARTs are free — do not use software serial.
- Sensor config as a table: `{uart, tx_pin, rx_pin, mount_angle_deg}`. **Write
  it for N sensors, not hardcoded 3.** I may run 2 during bringup.
- Avoid GPIO 33–37 if the board has octal PSRAM.
- Per-frame: validate header/tail, parse target slots, discard zeroed slots.
- Rotate each target from sensor-local X/Y into robot-frame `(bearing_deg,
  range_m)` using that sensor's mount angle.
- **Dedup across the seams.** A person near a 120° boundary will be reported by
  two sensors. Merge targets within a distance/bearing threshold into one.
- Emit a clean list of `(bearing_deg, range_m, confidence)` upstream at ~10 Hz.
  Confidence should fall off toward the ±60° edges of each sensor's FOV, where
  angle accuracy degrades toward the 20° end of spec.
- Fusion happens here, on the S3. The Mac receives meaning, not raw frames —
  same division of labor as the drive base.

### 2. Mac-side reader

- Match the USB device by **USB serial number** via
  `serial.tools.list_ports.comports()` → `.serial_number`, not by
  `/dev/cu.usbmodem*` path. Path numbering is unstable across reboots and
  replug order, and there are now two ESP32s attached. Pin the drive base the
  same way if it isn't already.
- Expose the target list to the rest of the system as its own source. **Do not
  fuse it into a shared occupancy grid with ToF.** They answer different
  questions: ToF is "is there a wall 40 cm ahead," radar is "is there a person
  at 4 m and 137°." Keep them as separate consumers of the spatial layer.
- Latch the last valid bearing for a few seconds on dropout. The LD2450 tracks
  moving targets and has no separate stationary channel, so a person who freezes
  can fall off the list. A dropout is not "nobody there."

### 3. Synthetic frame harness

- A generator that emits well-formed LD2450 frames for scripted target motion —
  single target crossing a seam, two targets, zero targets, malformed frame,
  mid-frame truncation.
- Must be able to drive the parser and the fusion logic with no hardware
  attached, so all of the above lands before the modules arrive.
- Keep this in the tree afterward; it is the regression test for the parser.

## Phasing

**Today (no hardware):** firmware structure, parser, rotation math, seam dedup,
Mac-side reader, synthetic harness. All testable against synthetic frames.

**Tomorrow (hardware):** flash, verify against real frames, check mutual
interference between adjacent free-running FMCW modules, validate PETG
penetration, tune the dedup thresholds and confidence falloff.

## Out of scope for this pass

- Do **not** wire radar into behavior selection yet. Land the data pipeline and
  let me see the bearings in logs first.
- Keep the existing haphazard scan intact as the fallback for when radar
  returns no targets.
- No changes to ToF handling.

> **Behavior wiring landed 2026-08-15** (`intelligence/motion_agency.py`,
> `MOTION_COME_RADAR_*` in `config.py`): the come-here search is radar-first.
> With no requester face on camera he turns straight to the best radar body
> (bearing = turn, at the scan rate), dwells for the camera, and if that body
> is not the requester (no face, or someone else's) marks the spot rejected —
> tracked in a world frame via the base's `imu.yaw` (commanded-turn sum as the
> fallback) — and turns to the next. Camera evidence always outranks radar (a
> visible/locked face → alignment; a fresh sighting → turn back first). Radar
> decisions use only ring frames received after a turn's `done` + settle, over
> a sample window, with a body required in several frames — the ring drifts
> while the base rotates, so turns are computed only when stationary. The
> haphazard sweep survives exactly as the fallback described above.

## Notes

- Power: sensors run off USB 5 V through the S3 board. ~80 mA each, ~240 mA
  total plus the S3 — inside a USB port's budget. Do not tap the robot's 5 V
  rail.
- Wire both TX and RX per sensor even though only RX is needed to read, so
  config can be pushed without pulling a module out of the ring.
- If a firmware crash takes the native USB CDC device down, the port vanishes
  from the Mac entirely rather than reconnecting. BOOT-button-while-plugging
  recovers it. Worth a line in the README.
