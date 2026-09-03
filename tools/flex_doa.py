#!/usr/bin/env python3
"""
tools/flex_doa.py — watch the reSpeaker Flex XVF3800's direction-of-arrival
over USB control (read-only, no audio capture, no writes).

    ./venv/bin/python tools/flex_doa.py                 # 20 s live monitor
    ./venv/bin/python tools/flex_doa.py --secs 60
    ./venv/bin/python tools/flex_doa.py --label "left"  # tag the run in the summary

Prints one line per poll while the chip flags speech (DOA_VALUE payload[1]),
then a summary: the circular median of the flagged DoA readings and how tightly
they cluster. Use it to establish the board's angle convention against the
robot's base frame (+ = left/CCW): stand at a known bearing, talk, read the
number. Stationary robot, one talker, no Rex playback.

Registers (per the XMOS XVF3800 command set):
  DOA_VALUE                 (deg 0-359, speech_detected 0/1) — the LED/DoA servicer
  AEC_AZIMUTH_VALUES        beam1, beam2, free-running, auto-select (radians)
  AEC_SPENERGY_VALUES       speech energy per beam (>0 = speech)
  AUDIO_MGR_SELECTED_AZIMUTHS  processed DoA (NaN when no fixed beam has speech),
                            auto-select beam DoA
"""

from __future__ import annotations

import argparse
import math
import sys
import time
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_ROOT))

from tools import flex_ctl  # noqa: E402


def circular_median_deg(angles: list[float]) -> float | None:
    """Median direction of a set of angles in degrees (0-359), or None if empty.

    Uses the angle that minimizes the summed circular distance to the others,
    so a cluster straddling 0/360 is handled and one wild reading can't pull it.
    """
    if not angles:
        return None
    best, best_cost = None, float("inf")
    for a in angles:
        cost = sum(abs((b - a + 180.0) % 360.0 - 180.0) for b in angles)
        if cost < best_cost:
            best, best_cost = a, cost
    return best


def circular_spread_deg(angles: list[float], center: float) -> float:
    """Mean absolute circular deviation from `center`, in degrees."""
    if not angles:
        return 0.0
    return sum(abs((b - center + 180.0) % 360.0 - 180.0) for b in angles) / len(angles)


def watch(secs: float, hz: float = 10.0, label: str = "") -> dict:
    dev = flex_ctl.open_device()
    flagged: list[float] = []
    beams: list[float] = []
    polls = 0
    period = 1.0 / max(1.0, hz)
    print(f"Flex DoA monitor — {secs:.0f}s at {hz:.0f} Hz. Talk from a known spot; "
          f"lines print only while the chip flags speech.")
    print(f"{'t':>5}  {'DoA':>4}  {'spk':>3}  {'auto-beam':>9}  {'spenergy(auto)':>14}")
    t0 = time.monotonic()
    with dev:
        while time.monotonic() - t0 < secs:
            t = time.monotonic()
            try:
                doa, spk = dev.read("DOA_VALUE")
                az = dev.read("AEC_AZIMUTH_VALUES")
                sp = dev.read("AEC_SPENERGY_VALUES")
            except Exception as exc:
                print(f"  read failed: {exc}")
                time.sleep(period)
                continue
            polls += 1
            auto_deg = (math.degrees(float(az[3])) % 360.0) if len(az) >= 4 else float("nan")
            if spk:
                flagged.append(float(doa))
                beams.append(auto_deg)
                print(f"{t - t0:5.1f}  {int(doa):4d}  {int(spk):3d}  {auto_deg:9.0f}  {float(sp[3]) if len(sp) >= 4 else float('nan'):14.2f}")
            time.sleep(max(0.0, period - (time.monotonic() - t)))

    med = circular_median_deg(flagged)
    out = {"label": label, "polls": polls, "speech_polls": len(flagged), "doa_median": med}
    print()
    if med is None:
        print(f"  {polls} polls, speech never flagged. Nobody talked, or the chip's VAD did "
              f"not trigger — try closer / louder, and check DOA_VALUE is live.")
        return out
    spread = circular_spread_deg(flagged, med)
    bmed = circular_median_deg(beams)
    out.update({"doa_spread": spread, "auto_beam_median": bmed})
    tag = f" [{label}]" if label else ""
    print(f"  {len(flagged)}/{polls} polls flagged speech{tag}: DoA median {med:.0f}° "
          f"(mean deviation {spread:.0f}°), auto-select beam median {bmed:.0f}°")
    if spread > 30.0:
        print("  ! readings are scattered — reflections, two talkers, or the robot moved.")
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--secs", type=float, default=20.0)
    ap.add_argument("--hz", type=float, default=10.0)
    ap.add_argument("--label", default="", help="tag for the summary line (e.g. 'left 90')")
    args = ap.parse_args()
    try:
        watch(args.secs, args.hz, args.label)
    except KeyboardInterrupt:
        print("\ninterrupted")
    except RuntimeError as exc:
        raise SystemExit(f"error: {exc}")


if __name__ == "__main__":
    main()
