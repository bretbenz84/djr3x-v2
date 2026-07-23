"""
hardware/compass.py — tilt-compensated magnetic heading + current-gated fusion.

The QMC5883L on the motion base publishes RAW axis counts in telemetry (`mag`
block, firmware/djr3x_motion/mag.cpp); this module turns them into a usable
heading. Three layers, each independently testable without hardware:

  1. Calibration (hard/soft-iron): per-axis offsets + scales measured in-situ
     by tools/compass_calibrate.py (figure-8 while mounted on R3X — the robot's
     own iron IS the thing being calibrated out). Persisted as JSON at
     config.COMPASS_CALIBRATION_PATH; loaded on init, loudly warned-about when
     absent. A simple min/max per-axis model — a full ellipsoid fit would be
     more accurate (soft-iron cross-axis terms) if we ever want it.

  2. Tilt compensation: project the calibrated field onto the horizontal plane
     using pitch/roll, then heading = atan2(-my_h, mx_h) + declination. In
     production pitch/roll come from the firmware IMU's complementary filter
     (telemetry `imu` block); the accel->pitch/roll helper here implements the
     same convention for tests and for callers holding a raw gravity vector.

  3. Current-gated fusion: complementary filter blending the (drift-prone but
     smooth) gyro yaw against the (absolute but load-corrupted) magnetic
     heading. The magnetometer trust weight is a function of |batt_ma| from the
     INA226 — the same telemetry field the power monitor already reads; no
     second reader. High current = motors working = magnetic garbage = gyro
     only. A magnitude gate additionally rejects samples whose |B| strays from
     the calibrated ambient field.

The telemetry source is injectable (any callable returning the motion telemetry
dict) so every layer runs against mocked data; the default source is
hardware.motion.telemetry — the single existing reader of the serial link.

Bench demo (robot on, Rex off):  venv/bin/python -m hardware.compass
Calibration:                     venv/bin/python tools/compass_calibrate.py
"""

from __future__ import annotations

import json
import logging
import math
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Optional

import config

_log = logging.getLogger(__name__)


# ── Calibration ────────────────────────────────────────────────────────────────

@dataclass
class Calibration:
    """Hard/soft-iron correction. offset = sample-cloud center (hard iron);
    scale = per-axis normalization (soft iron, axis-aligned approximation);
    field_norm = |B| of the calibrated ambient field (magnitude-gate reference)."""
    offset: tuple = (0.0, 0.0, 0.0)
    scale: tuple = (1.0, 1.0, 1.0)
    field_norm: float = 0.0
    loaded: bool = False          # False = identity fallback (uncalibrated!)

    def apply(self, x: float, y: float, z: float) -> tuple:
        ox, oy, oz = self.offset
        sx, sy, sz = self.scale
        return ((x - ox) * sx, (y - oy) * sy, (z - oz) * sz)


def load_calibration(path: "str | None" = None) -> Calibration:
    """Load the JSON produced by tools/compass_calibrate.py. Missing/corrupt
    file returns an identity calibration with loaded=False and a clear warning
    — headings still compute, but they are NOT trustworthy uncalibrated."""
    p = Path(path or getattr(config, "COMPASS_CALIBRATION_PATH", "compass_calibration.json"))
    if not p.is_absolute():
        p = Path(__file__).resolve().parent.parent / p
    try:
        d = json.loads(p.read_text())
        cal = Calibration(
            offset=tuple(float(v) for v in d["offset"]),
            scale=tuple(float(v) for v in d["scale"]),
            field_norm=float(d["field_norm"]),
            loaded=True,
        )
        _log.info("[compass] calibration loaded from %s (|B|=%.0f)", p, cal.field_norm)
        return cal
    except FileNotFoundError:
        _log.warning(
            "[compass] NO CALIBRATION at %s — headings are raw/untrustworthy. "
            "Run: venv/bin/python tools/compass_calibrate.py (in-situ, on the robot).", p
        )
    except Exception as exc:
        _log.warning("[compass] calibration unreadable (%s) — using identity. "
                     "Re-run tools/compass_calibrate.py.", exc)
    return Calibration()


def save_calibration(cal: Calibration, path: "str | None" = None) -> Path:
    p = Path(path or getattr(config, "COMPASS_CALIBRATION_PATH", "compass_calibration.json"))
    if not p.is_absolute():
        p = Path(__file__).resolve().parent.parent / p
    p.write_text(json.dumps(
        {"offset": list(cal.offset), "scale": list(cal.scale),
         "field_norm": cal.field_norm, "saved_at": time.strftime("%Y-%m-%dT%H:%M:%S")},
        indent=2,
    ) + "\n")
    return p


def compute_calibration(samples: "list[tuple]") -> Calibration:
    """Hard iron = center of the min/max box per axis; soft iron = per-axis
    scale equalizing the three half-ranges (axis-aligned approximation — a full
    ellipsoid least-squares fit would also capture cross-axis skew; upgrade
    here if headings stay lumpy after a good figure-8)."""
    if len(samples) < 10:
        raise ValueError(f"need at least 10 samples, got {len(samples)}")
    xs, ys, zs = zip(*samples)
    mins = (min(xs), min(ys), min(zs))
    maxs = (max(xs), max(ys), max(zs))
    offset = tuple((mx + mn) / 2.0 for mn, mx in zip(mins, maxs))
    half = [(mx - mn) / 2.0 for mn, mx in zip(mins, maxs)]
    if min(half) <= 0:
        raise ValueError("degenerate sample cloud (an axis never varied) — rotate through a full figure-8")
    avg = sum(half) / 3.0
    scale = tuple(avg / h for h in half)
    cal = Calibration(offset=offset, scale=scale, loaded=True)
    # Ambient |B| reference: mean calibrated magnitude over the cloud.
    mags = [math.sqrt(sum(c * c for c in cal.apply(*s))) for s in samples]
    cal.field_norm = sum(mags) / len(mags)
    return cal


# ── Pure math (unit-tested against known vectors) ──────────────────────────────

def accel_to_pitch_roll(ax: float, ay: float, az: float) -> tuple:
    """Gravity direction -> (pitch, roll) in radians.

    Convention (matches the firmware IMU's accel seed in imu.cpp):
      pitch = asin(-ax_n)          — nose down = ax positive = negative pitch
      roll  = asin(ay_n / cos(pitch))
    Guards: normalizes the input, clamps asin domains, and pins roll to 0 when
    cos(pitch) ~ 0 (pointing straight up/down — roll is undefined there).
    """
    n = math.sqrt(ax * ax + ay * ay + az * az)
    if n < 1e-9:
        return 0.0, 0.0
    axn, ayn = ax / n, ay / n
    pitch = math.asin(max(-1.0, min(1.0, -axn)))
    cp = math.cos(pitch)
    if abs(cp) < 1e-6:
        return pitch, 0.0
    roll = math.asin(max(-1.0, min(1.0, ayn / cp)))
    return pitch, roll


def tilt_compensated_heading(mx: float, my: float, mz: float,
                             pitch: float, roll: float,
                             declination_deg: float = 0.0) -> float:
    """Project the (calibrated) field onto the horizontal plane and return
    heading in degrees [0, 360). pitch/roll in RADIANS. Standard AN4248-style
    de-rotation; the axis-sign FLIP/SWAP flags in config are applied by the
    caller (Compass._mapped_mag), not here — this stays pure math."""
    sp, cp = math.sin(pitch), math.cos(pitch)
    sr, cr = math.sin(roll), math.cos(roll)
    mx_h = mx * cp + mz * sp
    my_h = mx * sr * sp + my * cr - mz * sr * cp
    heading = math.degrees(math.atan2(-my_h, mx_h)) + declination_deg
    return heading % 360.0


def ang_diff(a: float, b: float) -> float:
    """Shortest signed angular difference a-b in degrees, in (-180, 180]."""
    d = (a - b) % 360.0
    return d - 360.0 if d > 180.0 else d


def blend_heading(current: float, target: float, alpha: float) -> float:
    """Pull `current` toward `target` by fraction alpha along the SHORTEST arc
    (359° blended toward 1° moves through 0°, not backward through 180°)."""
    return (current + alpha * ang_diff(target, current)) % 360.0


def alpha_for_current(ma: "float | None") -> float:
    """Magnetometer trust weight as a function of |pack current| (mA).
    <= LOW -> ALPHA_MAX (idle, field clean); >= HIGH -> ALPHA_MIN (motors
    working, field garbage); linear ramp between. Unknown current -> distrust
    (ALPHA_MIN) — the safe direction."""
    lo = float(getattr(config, "COMPASS_CURRENT_LOW_MA", 1600))
    hi = float(getattr(config, "COMPASS_CURRENT_HIGH_MA", 2600))
    a_max = float(getattr(config, "COMPASS_ALPHA_MAX", 0.05))
    a_min = float(getattr(config, "COMPASS_ALPHA_MIN", 0.0))
    if ma is None:
        return a_min
    cur = abs(float(ma))
    if cur <= lo:
        return a_max
    if cur >= hi or hi <= lo:
        return a_min
    frac = (cur - lo) / (hi - lo)
    return a_max + (a_min - a_max) * frac


def field_magnitude_ok(mx: float, my: float, mz: float, cal: Calibration,
                       tolerance: "float | None" = None) -> bool:
    """Magnitude sanity gate: a calibrated sample whose |B| strays beyond
    tolerance (fraction) of the calibrated ambient field is contaminated
    (motor transient, nearby magnet) and must not steer the fusion. Passes
    everything when uncalibrated (no reference to gate against)."""
    if not cal.loaded or cal.field_norm <= 0:
        return True
    tol = float(tolerance if tolerance is not None
                else getattr(config, "COMPASS_FIELD_TOLERANCE", 0.25))
    mag = math.sqrt(mx * mx + my * my + mz * mz)
    return abs(mag - cal.field_norm) <= tol * cal.field_norm


# ── The compass ────────────────────────────────────────────────────────────────

class Compass:
    """Heading + fusion over the motion telemetry stream.

    telemetry_source: callable returning the motion telemetry dict (or None).
    Defaults to hardware.motion.telemetry — inject a fake for tests/benches.
    """

    def __init__(self, telemetry_source: "Callable[[], Optional[dict]] | None" = None,
                 calibration: "Calibration | None" = None,
                 calibration_path: "str | None" = None):
        if telemetry_source is None:
            from hardware import motion
            telemetry_source = motion.telemetry
        self._source = telemetry_source
        self.cal = calibration if calibration is not None else load_calibration(calibration_path)
        self._heading: "float | None" = None     # last tilt-compensated heading (deg true)
        self._fused: "float | None" = None       # fused yaw estimate (deg true)
        self._pitch = 0.0                        # last pitch/roll used (deg)
        self._roll = 0.0
        self._alpha = 0.0                        # last trust weight applied
        self._prev_gyro_yaw: "float | None" = None
        self._rejected = 0                       # magnitude-gate rejections
        self._updates = 0

    # ── one fusion step ─────────────────────────────────────────────────────────
    def update(self) -> None:
        """Consume one telemetry snapshot: recompute the tilt-compensated
        heading, advance the fused yaw by the gyro delta, and (gated by current
        and |B|) pull it toward the magnetic heading. Call at ~10 Hz."""
        snap = self._source() or {}
        mag = snap.get("mag") or {}
        imu = snap.get("imu") or {}
        ma = snap.get("batt_ma")

        # Gyro increment: the firmware IMU's yaw IS the bias-corrected gyro
        # integral (imu.cpp) — successive deltas are the yaw-rate * dt term.
        gyro_yaw = imu.get("yaw") if imu.get("ok") else None
        if gyro_yaw is not None and self._prev_gyro_yaw is not None and self._fused is not None:
            self._fused = (self._fused + ang_diff(float(gyro_yaw), self._prev_gyro_yaw)) % 360.0
        self._prev_gyro_yaw = float(gyro_yaw) if gyro_yaw is not None else self._prev_gyro_yaw

        if not mag.get("ok") or mag.get("ovl"):
            self._alpha = 0.0
            return
        mx, my, mz = self._mapped_mag(mag)
        mx, my, mz = self.cal.apply(mx, my, mz)

        pitch_deg = float(imu.get("pitch", 0.0) or 0.0)
        roll_deg = float(imu.get("roll", 0.0) or 0.0)
        self._pitch, self._roll = pitch_deg, roll_deg
        heading = tilt_compensated_heading(
            mx, my, mz, math.radians(pitch_deg), math.radians(roll_deg),
            declination_deg=float(getattr(config, "COMPASS_DECLINATION_DEG", 0.0)),
        )
        self._heading = heading
        self._updates += 1

        if not field_magnitude_ok(mx, my, mz, self.cal):
            self._rejected += 1
            self._alpha = 0.0
            return

        self._alpha = alpha_for_current(ma)
        if self._fused is None:
            self._fused = heading                # first anchor: take it outright
        else:
            self._fused = blend_heading(self._fused, heading, self._alpha)

    # ── axis mapping (⚠ mounting not finalized — config flags, not math edits) ──
    @staticmethod
    def _mapped_mag(mag: dict) -> tuple:
        x, y, z = float(mag.get("x", 0)), float(mag.get("y", 0)), float(mag.get("z", 0))
        if getattr(config, "COMPASS_SWAP_XY", False):
            x, y = y, x
        if getattr(config, "COMPASS_FLIP_X", False):
            x = -x
        if getattr(config, "COMPASS_FLIP_Y", False):
            y = -y
        if getattr(config, "COMPASS_FLIP_Z", False):
            z = -z
        return x, y, z

    # ── API ─────────────────────────────────────────────────────────────────────
    def get_heading(self) -> "float | None":
        """Latest raw tilt-compensated magnetic heading (deg true, 0-360),
        None until the first valid magnetometer sample."""
        return self._heading

    def get_fused_yaw(self) -> "float | None":
        """The current-gated gyro+mag fusion output (deg true, 0-360)."""
        return self._fused

    def status(self) -> dict:
        return {
            "calibrated": self.cal.loaded,
            "field_norm": self.cal.field_norm,
            "heading": self._heading,
            "fused_yaw": self._fused,
            "pitch": self._pitch,
            "roll": self._roll,
            "alpha": self._alpha,
            "rejected": self._rejected,
            "updates": self._updates,
        }


# ── Background service (COMPASS_ENABLED; the QMC isn't wired yet — scaffold) ──
# main.py starts this once the magnetometer is physically on the trunk and
# COMPASS_ENABLED is flipped. Consumers (room-model heading tags, exploration
# leg planning) read get_service_yaw(); None until the service runs AND the
# sensor answers. Spatial anchoring proper (landmarks at headings) builds on
# this — deferred until the hardware exists (see context.md).

_service: "Compass | None" = None
_service_thread = None


def start_service(hz: float = 10.0) -> bool:
    """Idempotent background fusion loop over the motion telemetry stream."""
    global _service, _service_thread
    import threading
    if not bool(getattr(config, "COMPASS_ENABLED", False)):
        return False
    if _service_thread is not None:
        return True
    _service = Compass()

    def _loop():
        import time as _t
        while True:
            try:
                _service.update()
            except Exception:
                pass
            _t.sleep(1.0 / hz)

    _service_thread = threading.Thread(target=_loop, daemon=True, name="compass-service")
    _service_thread.start()
    _log.info("[compass] service started (%.0f Hz)%s", hz,
              "" if _service.cal.loaded else " — UNCALIBRATED")
    return True


def get_service_yaw(*, require_calibrated: bool = False) -> "float | None":
    """Fused true heading, or None when unavailable.

    Motion verification must pass ``require_calibrated=True``: an uncalibrated
    magnetometer can look numerically valid while being badly distorted by the
    chassis, so it is suitable for diagnostics but never closed-loop correction.
    """
    if _service is None:
        return None
    if require_calibrated and not _service.cal.loaded:
        return None
    return _service.get_fused_yaw()


def service_calibrated() -> bool:
    """Whether the running service has a real hard/soft-iron calibration."""
    return bool(_service is not None and _service.cal.loaded)


# ── Bench demo loop ────────────────────────────────────────────────────────────

def demo(hz: float = 10.0) -> int:
    """Print heading / fused yaw / pitch / roll / trust at ~10 Hz. Needs the
    base streaming telemetry (Rex off; the battery menubar must not hold the
    port — hardware.motion owns it here)."""
    from hardware import motion
    if not motion.connect():
        print("Could not open the motion serial link (is Rex or the battery "
              "menubar holding the port?)")
        return 1
    c = Compass()
    if not c.cal.loaded:
        print("⚠  UNCALIBRATED — headings below are raw. Run tools/compass_calibrate.py first.")
    try:
        while True:
            c.update()
            s = c.status()
            hdg = f"{s['heading']:6.1f}°" if s["heading"] is not None else "  --  "
            fus = f"{s['fused_yaw']:6.1f}°" if s["fused_yaw"] is not None else "  --  "
            print(f"heading={hdg}  fused={fus}  pitch={s['pitch']:+5.1f}°  "
                  f"roll={s['roll']:+5.1f}°  alpha={s['alpha']:.3f}  "
                  f"rejected={s['rejected']}", flush=True)
            time.sleep(1.0 / hz)
    except KeyboardInterrupt:
        pass
    finally:
        motion.disconnect()
    return 0


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(name)s | %(message)s")
    raise SystemExit(demo())
