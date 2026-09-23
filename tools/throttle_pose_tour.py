#!/usr/bin/env python3
"""Supervised tour of owner-recorded poses. CLI defaults to a read-only plan.

Requires clearance and an observer. This uses a raised-shoulder transition strategy,
not geometric collision certification. Only channels 8, 9 and 10 are commanded.
"""
import argparse
import csv
import json
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
from tools import rex_servo_menubar as helper
from tools.throttle_sequences import validate_pose
from utils import single_instance
from hardware.throttle_motion import remember_park

# Actual Maestro endpoints observed in the owner's recorded measurements.
RAISED = 544 * 4
PARK = {8: 2272 * 4, 9: 2496 * 4, 10: 512 * 4}
CHANNELS = (8, 9, 10)
STOP_FILE = ROOT / 'data' / 'throttle-tour.stop'


def load_poses():
    with (ROOT / 'data' / 'throttle_measurements.csv').open(newline='') as stream:
        rows = list(csv.DictReader(stream))
    poses = [dict(id=i + 1, note=row['note'], target={
        ch: round(float(row[key]) * 4) for ch, key in
        ((8, 'shoulder_us'), (9, 'elbow_us'), (10, 'wrist_us'))})
        for i, row in enumerate(rows)]
    # Work through the high-shoulder poses first, then intermediate, then lowered.
    return sorted(poses, key=lambda pose: (pose['target'][8], pose['id']))



def elbow_minimum(shoulder):
    # Conservative bounds between the owner's measured shoulder positions.
    if shoulder > 1702 * 4:
        return 1546 * 4
    if shoulder > 1636 * 4:
        return 636 * 4
    return 500 * 4


def respects_elbow_limit(pose):
    return pose[9] >= elbow_minimum(pose[8])


def transition(current, target):
    """Raise alone, position both downstream joints, then lower alone."""
    stages = []
    if current == target:
        return stages
    if current[8] != RAISED:
        stages.append({8: RAISED})
    downstream = {ch: target[ch] for ch in (9, 10) if current[ch] != target[ch]}
    if downstream:
        stages.append(downstream)
    if target[8] != RAISED:
        stages.append({8: target[8]})
    return stages


def check_stop():
    if STOP_FILE.exists():
        raise InterruptedError('Stop file requested hold')


def wait_seconds(seconds):
    until = time.monotonic() + seconds
    while time.monotonic() < until:
        check_stop()
        time.sleep(min(.05, max(0, until - time.monotonic())))


class Controller:
    def __init__(self, port, limits, recorded_poses=(), settle=.35):
        self.port, self.limits = port, limits
        self.settle = settle
        # Explicit supervised recorded-pose mode permits only the saved tuples,
        # not a globally lowered elbow minimum or arbitrary interpolated poses.
        self.recorded_poses = tuple(dict(pose) for pose in recorded_poses)

    def allowed(self, pose):
        return respects_elbow_limit(pose) or any(
            all(abs(pose[ch] - saved[ch]) <= 2 for ch in CHANNELS)
            for saved in self.recorded_poses)

    def read(self):
        pose = helper._read_positions(self.port, list(CHANNELS))
        validate_pose(pose, self.limits)
        return pose

    def move(self, targets):
        check_stop()
        if not set(targets) <= set(CHANNELS):
            raise ValueError('Only throttle targets are allowed')
        predicted = self.read()
        predicted.update(targets)
        if not self.allowed(predicted):
            raise ValueError('Pose conflicts with the established shoulder/elbow limit')
        for ch, value in targets.items():
            cfg = self.limits[ch]
            if not cfg['min'] <= value <= cfg['max']:
                raise ValueError(f'Invalid target on channel {ch}')
            helper._write_target(self.port, cfg, value)
        deadline = time.monotonic() + 20
        stable = 0
        while time.monotonic() < deadline:
            check_stop()
            current = self.read()
            if all(abs(current[ch] - value) <= 2 for ch, value in targets.items()):
                stable += 1
                if stable >= (1 if self.settle == 0 else 3):
                    wait_seconds(self.settle)  # Extra physical settling; pulses are not shaft feedback.
                    return
            else:
                stable = 0
            wait_seconds(.1)
        raise TimeoutError('Throttle did not reach requested pulse targets within 20 seconds')

    def reach(self, target):
        for stage in transition(self.read(), target):
            print('  Move', {ch: value / 4 for ch, value in stage.items()}, flush=True)
            self.move(stage)
        result = self.read()
        if any(abs(result[ch] - target[ch]) > 2 for ch in CHANNELS):
            raise ValueError('Pose readback mismatch')
        print('  Reached (pulse µs)', {ch: result[ch] / 4 for ch in CHANNELS}, flush=True)

    def hold(self):
        helper._hold_throttle(self.port, self.limits)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--action', choices=('plan', 'inspect', 'park', 'run'), default='plan')
    parser.add_argument('--limits', choices=('established', 'recorded'), default='established',
                        help='Recorded mode explicitly permits exact owner-recorded poses for a supervised tour')
    parser.add_argument('--dwell', type=float, default=3)
    parser.add_argument('--settle', type=float, default=.35)
    args = parser.parse_args()
    if not 0 <= args.dwell <= 60 or not 0 <= args.settle <= 5:
        parser.error('--dwell must be 0–60 and --settle 0–5 seconds')
    limits = {cfg['ch']: cfg for cfg in helper._servos().values()}
    poses = load_poses()
    for pose in poses:
        validate_pose(pose['target'], limits)
    recorded_poses = [pose['target'] for pose in poses] if args.limits == 'recorded' else []
    skipped = [pose for pose in poses if not respects_elbow_limit(pose['target'])] if args.limits == 'established' else []
    poses = [pose for pose in poses if pose not in skipped]
    if args.action == 'plan':
        print(json.dumps(dict(dwell_seconds=args.dwell, limits=args.limits, park=PARK, poses=poses, skipped_conflicting_poses=skipped), indent=2))
        return
    check_stop()
    if not single_instance.acquire():
        raise RuntimeError('Rex is running; no connection or movement attempted')
    try:
        # Holding Rex's lock makes the menu helper release its serial connection.
        time.sleep(1.5)
        import serial
        with serial.Serial(helper._maestro_port(), 9600, timeout=.3, exclusive=True) as port:
            controller = Controller(port, limits, recorded_poses, args.settle)
            print('Current pulse µs', {ch: value / 4 for ch, value in controller.read().items()}, flush=True)
            helper._require_stationary(port)
            if args.action == 'inspect':
                if controller.read() == PARK:
                    remember_park()
                return
            try:
                print('Parking throttle arm', flush=True)
                controller.reach(PARK)
                if args.action == 'run':
                    print('Omitting poses conflicting with elbow limit:', [pose['id'] for pose in skipped], flush=True)
                    wait_seconds(args.dwell)
                    for step, pose in enumerate(poses, 1):
                        print(f"Step {step}/{len(poses)}, saved pose {pose['id']}: {pose['note']}", flush=True)
                        controller.reach(pose['target'])
                        print(f'  Hold {args.dwell:g} seconds', flush=True)
                        wait_seconds(args.dwell)
                    print('Returning to parked pose', flush=True)
                    controller.reach(PARK)
                print('DONE — parked', flush=True)
                remember_park()
            except BaseException:
                print('ABORT — holding current pulses; no automatic return move', flush=True)
                controller.hold()
                raise
    finally:
        single_instance.release()


if __name__ == '__main__':
    main()
