# DJ-R3X Control Dashboard

The dashboard is an optional PySide6 / Qt UI for watching Rex while the normal
controller owns the hardware, audio, vision, memory, and consciousness loops.
It is disabled by default and should not change headless behavior.

## Enable

In `config.py`:

```python
GUI_ENABLED = True
GUI_BACKEND = "pyside6"
```

Then run Rex normally from the project venv:

```bash
source venv/bin/activate
python main.py
```

If PySide6 or a display is unavailable, `main.py` logs a warning and continues
headless.

## Dependencies

`PySide6` is installed by `pip install -r requirements.txt`. No Homebrew GUI
package is required.

## Demo Mode

Run the dashboard without the controller or hardware:

```bash
source venv/bin/activate
python -m gui.dashboard --demo
```

Demo mode generates a placeholder camera frame, fake people boxes, conversation
lines, and simulated servo motion.

## Motivator Control

The 🕹 MOTIVATOR button (servo panel) opens a four-column drive console:
joystick + commanded output | radar-ring scope | ToF (photoreceptors + 8×8
matrix) | gamepad mirror, attitude, and raw ESP32 feedback.

The **RADAR RING** panel is a top-down scope of the LD2450 bearing-prior ring,
polled from `hardware.radar` on the telemetry tick: front is up, + bearing =
left/CCW, dashed rings every 2 m to the 8 m full scale. Each sensor's ±60° FOV
wedge (mounts from the board's hello) turns red when that sensor stops
delivering; fused targets plot at (bearing, range) as confidence-colored dots
(green ≥ 0.55, amber ≥ 0.30, gray below) with a radial speed tail and a white
halo when two sensors agreed across a seam. An empty live frame with
`radar.targets()` still latching draws hollow dots under a LATCHED chip —
"remembering", not "seeing". Per-sensor frames/bad/drop counters line the
footer. The drive-base and radar links are independent — either panel stays
live while the other board is down.

## How State Reaches Qt

The GUI reads only copied data from `gui.state_bridge.gui_bridge`.

- `main.py` starts a small bridge-sync thread in GUI mode.
- The sync thread copies `vision.camera.get_frame()` and `world_state.snapshot()`.
- `utils.conv_log` mirrors transcribed user lines and Rex speech into the bridge.
- `hardware.servos` mirrors commanded servo targets into `world_state` and the
  bridge so the avatar can move in software-only mode.
- Qt polls the bridge with a `QTimer` at `GUI_FPS`.

The GUI does not open a camera, microphone, serial device, audio stream, or LLM
connection.

## Known Limitations

- The avatar is a simple 2D visualization, not a mechanically exact CAD model.
- Person boxes depend on `world_state.people[*].face_box`; if only a point is
  available, the vision panel draws a marker instead.
- Closing the GUI requests normal `SHUTDOWN`; shutdown audio/servo animation may
  continue briefly while the controller finishes cleanup.
