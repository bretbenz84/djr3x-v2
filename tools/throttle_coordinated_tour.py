#!/usr/bin/env python3
"""Supervised coordinated version of the owner's tested throttle tour.

Moves shoulder with downstream joints only inside conservative clearance boxes.
No arbitrary interpolation between the two low-elbow recorded exceptions.
"""
import argparse
import json
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
from tools import throttle_pose_tour as tour
from tools import rex_servo_menubar as helper
from tools.throttle_sequences import validate_pose
from utils import single_instance
from hardware.throttle_motion import clearance_box, invalidate_park, remember_park


def stages(start, end):
    if start == end:
        return []
    if clearance_box(start, end):
        return [dict(end)]
    current = dict(start)
    result = []

    def add(pose):
        nonlocal current
        if pose != current:
            result.append(dict(pose))
            current = dict(pose)

    # Clear the wrist while raised, then lower shoulder and extend elbow together
    # toward the measured intermediate pose (no extra clearance assumed).
    if current[8] == tour.RAISED and end[8] <= 1384.5 * 4:
        folded = dict(current)
        folded[10] = min(current[10], end[10], 2254.25 * 4)
        if clearance_box(folded, end):
            add(folded)
            add(end)
            return result

    if current[8] > 1636 * 4:
        gate = dict(current)
        gate[8] = 1636 * 4
        if current[10] <= 512 * 4 and current[9] >= 1546 * 4:
            gate[9] = max(1546 * 4, end[9])
        add(gate)
    raised_destination = dict(end)
    raised_destination[8] = tour.RAISED
    if clearance_box(current, raised_destination):
        add(raised_destination)
    else:
        high = dict(current)
        high[8] = tour.RAISED
        add(high)
        add(raised_destination)
    add(end)
    return result


def make_plan():
    current = dict(tour.PARK)
    plan = []
    for pose in tour.load_poses() + [dict(id='park', target=tour.PARK)]:
        moves = stages(current, pose['target'])
        for index, end in enumerate(moves):
            changing = [ch for ch in tour.CHANNELS if current[ch] != end[ch]]
            combined = 8 in changing and len(changing) > 1
            if combined and not clearance_box(current, end):
                raise ValueError('Shoulder combination outside clearance model')
            plan.append(dict(start=dict(current), end=dict(end), combined=combined,
                             pose_id=pose['id'] if index == len(moves)-1 else None))
            current = dict(end)
    return plan


# Gesture intentions, not Cartesian coordinates: geometry has not been calibrated.
# Each reach lifts the shoulder (smaller pulse), opens the elbow (smaller pulse),
# and uncurls the wrist toward approximately straight (1500 us).
REACH_STUDY_US = (
    ('ready', 1636, 1700, 512),
    ('curious-reach', 1280, 1250, 1150),
    ('reach-forward', 760, 700, 1485),
    ('draw-back', 1420, 1600, 1050),
    ('offer', 1050, 1100, 1320),
    ('extend-offer', 600, 780, 1500),
    ('gather', 1300, 1650, 1050),
    ('reach-higher', 700, 1100, 1550),
    ('relax', 1500, 1800, 850),
    ('tuck', 1636, 2100, 512),
    ('park', 2272, 2496, 512),
)


# Continue into upward reaches that sweep forward before drawing back in.
# High elbow pulses use the owner's "forearm straight up" measured orientation;
# varied wrist angles make each sweep unfurl or curl the hand with the arm.
REACH_AND_UP_US = REACH_STUDY_US[:-3] + (
    ('prepare-upward', 1050, 1800, 900),
    ('reach-up-one', 560, 2250, 1950),
    ('sweep-forward-one', 850, 700, 1485),
    ('draw-in-one', 1220, 1680, 850),
    ('reach-up-two', 650, 2320, 1150),
    ('sweep-forward-two', 900, 780, 1550),
    ('draw-in-two', 1250, 1750, 950),
    ('reach-up-three', 580, 2340, 1850),
    ('sweep-forward-three', 760, 650.25, 1484.5),
) + REACH_STUDY_US[-3:]


def make_reach_plan(poses=REACH_STUDY_US):
    """Three expressive reaches; every move combines shoulder and elbow.

    No standalone wrist beats, full-extension rails, or low-elbow exceptions.
    New intermediate poses use the existing conservative clearance-box model.
    """
    current = dict(tour.PARK)
    plan = []
    for name, shoulder, elbow, wrist in poses:
        end = {8: round(shoulder * 4), 9: round(elbow * 4), 10: round(wrist * 4)}
        if not clearance_box(current, end):
            raise ValueError(f'Reach-study transition outside clearance model: {name}')
        if current[8] == end[8] or current[9] == end[9]:
            raise ValueError(f'Reach-study movement must combine shoulder and elbow: {name}')
        plan.append(dict(start=dict(current), end=end, combined=True, pose_id=name))
        current = end
    return plan


class CoordinatedController(tour.Controller):
    def move_pose(self, end):
        tour.check_stop()
        current = self.read()
        validate_pose(end, self.limits)
        changing = [ch for ch in tour.CHANNELS if abs(current[ch] - end[ch]) > 2]
        if 8 in changing and len(changing) > 1 and not clearance_box(current, end):
            raise ValueError('Combined shoulder move outside clearance box')
        if not self.allowed(end):
            raise ValueError('Target outside established or recorded pose limits')
        distances = {ch: abs(end[ch] - current[ch]) for ch in changing}
        if not distances:
            return
        invalidate_park()
        # Speed and acceleration proportional to travel approximate a common
        # arrival time. Clearance box tolerates different actual progress rates.
        speed_scale = max(distances[ch] / self.limits[ch]['speed'] for ch in changing)
        accel_scale = max(distances[ch] / self.limits[ch]['acceleration'] for ch in changing)
        for ch in changing:
            speed = max(1, min(self.limits[ch]['speed'], round(distances[ch]/speed_scale)))
            acceleration = max(1, min(self.limits[ch]['acceleration'], round(distances[ch]/accel_scale)))
            for cmd, value in ((0x89, acceleration), (0x87, speed)):
                packet = bytes([cmd, ch, value & 127, value >> 7])
                if self.port.write(packet) != len(packet):
                    raise OSError('Incomplete profile write')
        packet = bytearray([0x9F, 3, 8])
        for ch in tour.CHANNELS:
            packet.extend((end[ch] & 127, end[ch] >> 7))
        if self.port.write(packet) != len(packet):
            raise OSError('Incomplete simultaneous target write')
        deadline = time.monotonic() + 30
        while time.monotonic() < deadline:
            tour.check_stop()
            actual = self.read()
            if all(abs(actual[ch]-end[ch]) <= 2 for ch in tour.CHANNELS):
                return
            tour.wait_seconds(.05)
        raise TimeoutError('Coordinated movement did not arrive')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run', action='store_true')
    parser.add_argument('--routine', choices=('pose-tour', 'reach-study', 'reach-and-up'), default='pose-tour')
    args = parser.parse_args()
    limits = {cfg['ch']: cfg for cfg in helper._servos().values()}
    if args.routine == 'reach-and-up':
        plan = make_reach_plan(REACH_AND_UP_US)
    else:
        plan = make_reach_plan() if args.routine == 'reach-study' else make_plan()
    for move in plan:
        validate_pose(move['end'], limits)
    if not args.run:
        print(json.dumps(dict(routine=args.routine, dwell_seconds=0, combined_shoulder_moves=sum(m['combined'] for m in plan), moves=plan), indent=2))
        return
    tour.check_stop()
    if not single_instance.acquire():
        raise RuntimeError('Rex is running; did not open Maestro')
    try:
        time.sleep(1.5)
        import serial
        with serial.Serial(helper._maestro_port(),9600,timeout=.3,exclusive=True) as port:
            controller = CoordinatedController(port, limits, [p['target'] for p in tour.load_poses()], settle=0)
            start = controller.read()
            if any(abs(start[ch]-tour.PARK[ch]) > 2 for ch in tour.CHANNELS):
                raise ValueError('Arm must start parked; no movement sent')
            helper._require_stationary(port)
            try:
                for move in plan:
                    label = 'COMBINED shoulder + downstream' if move['combined'] else 'Transition'
                    print(label, {ch: v/4 for ch,v in move['end'].items()}, flush=True)
                    controller.move_pose(move['end'])
                    if move['pose_id'] is not None:
                        print(f"Reached pose {move['pose_id']} — no dwell",flush=True)
                print('DONE — parked', {ch:v/4 for ch,v in controller.read().items()},flush=True)
                remember_park()
            except BaseException:
                print('ABORT — holding current pulses',flush=True)
                controller.hold()
                raise
    finally:
        single_instance.release()


if __name__ == '__main__':
    main()
