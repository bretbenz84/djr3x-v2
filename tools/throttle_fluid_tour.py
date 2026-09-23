#!/usr/bin/env python3
"""No-dwell coordinated replay of the owner's collision-free raised-shoulder route.

This preserves tested transition paths; it is not an arbitrary collision planner.
Default action writes only a plan. --run commands the real throttle arm.
"""
import argparse
import json
import math
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

# Below the existing Maestro speed caps (30/70/70), in microseconds/second.
VELOCITY_US = {8: 500, 9: 1000, 10: 1000}
# Pulse-trajectory acceleration budgets (microseconds/second²).
ACCELERATION_US = {8: 900, 9: 2200, 10: 2200}
PERIOD = .02


def ease(u):
    return u * u * u * (10 + u * (-15 + 6 * u))


def duration(start, end):
    # Quintic smoothstep peaks: velocity 1.875, acceleration 10/sqrt(3).
    return max(.35, *(max(1.875 * abs(end[ch] - start[ch]) / 4 / VELOCITY_US[ch],
                          math.sqrt((10 / math.sqrt(3)) * abs(end[ch] - start[ch]) / 4 / ACCELERATION_US[ch]))
                       for ch in tour.CHANNELS))


def interpolate(start, end, u):
    weight = ease(max(0, min(1, u)))
    return {ch: round(start[ch] + (end[ch] - start[ch]) * weight) for ch in tour.CHANNELS}


def build_plan():
    current = dict(tour.PARK)
    segments = []
    poses = tour.load_poses()
    for pose in poses + [dict(id='park', note='Park', target=tour.PARK)]:
        stages = tour.transition(current, pose['target'])
        for i, stage in enumerate(stages):
            end = dict(current)
            end.update(stage)
            segments.append(dict(start=dict(current), end=end, seconds=duration(current, end),
                                 pose_id=pose['id'] if i == len(stages) - 1 else None,
                                 note=pose['note']))
            current = end
    return segments


def verify_plan(segments, limits):
    current = tour.PARK
    visited = []
    for segment in segments:
        if segment['start'] != current:
            raise ValueError('Discontinuous trajectory')
        start, end = segment['start'], segment['end']
        changing = [ch for ch in tour.CHANNELS if start[ch] != end[ch]]
        # Preserve the tested corridor exactly: shoulder alone or downstream
        # changes with the shoulder fully raised. No corner-cutting blends.
        if 8 in changing:
            if changing != [8]:
                raise ValueError('Untested shoulder/downstream simultaneous motion')
        elif start[8] != tour.RAISED or end[8] != tour.RAISED:
            raise ValueError('Downstream motion must use the verified raised corridor')
        for i in range(math.ceil(segment['seconds'] / PERIOD) + 1):
            point = interpolate(start, end, min(1, i * PERIOD / segment['seconds']))
            validate_pose(point, limits)
            for ch in tour.CHANNELS:
                if not min(start[ch], end[ch]) <= point[ch] <= max(start[ch], end[ch]):
                    raise ValueError('Trajectory overshoot')
        current = end
        if segment['pose_id'] is not None:
            visited.append(segment['pose_id'])
    if set(visited) != {pose['id'] for pose in tour.load_poses()} | {'park'} or current != tour.PARK:
        raise ValueError('Tour must visit every saved pose and finish parked')


def send_frame(port, pose):
    from hardware.throttle_motion import invalidate_park
    invalidate_park()
    # Mini Maestro Set Multiple Targets: contiguous channels 8, 9, 10.
    packet = bytearray([0x9F, 3, 8])
    for ch in tour.CHANNELS:
        value = pose[ch]
        packet.extend((value & 0x7F, (value >> 7) & 0x7F))
    if port.write(packet) != len(packet):
        raise OSError('Incomplete coordinated target write')


def current_pose(port, limits):
    pose = helper._read_positions(port, list(tour.CHANNELS))
    validate_pose(pose, limits)
    return pose


def wait_arrival(port, target, limits, timeout=5):
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        tour.check_stop()
        pose = current_pose(port, limits)
        if all(abs(pose[ch] - target[ch]) <= 2 for ch in tour.CHANNELS):
            return pose
        tour.wait_seconds(.02)
    raise TimeoutError('Pulse trajectory failed to arrive; stopping')


def run(port, plan, limits):
    start = current_pose(port, limits)
    if any(abs(start[ch] - tour.PARK[ch]) > 2 for ch in tour.CHANNELS):
        raise ValueError('Start must be parked; no automatic repositioning')
    helper._require_stationary(port)
    for ch in tour.CHANNELS:
        cfg = limits[ch]
        for command, value in ((0x89, cfg['acceleration']), (0x87, cfg['speed'])):
            packet = bytes([command, ch, value & 0x7F, (value >> 7) & 0x7F])
            if port.write(packet) != len(packet):
                raise OSError('Could not set motion profile')
    started = time.monotonic()
    max_lag = 0
    for segment in plan:
        frames = math.ceil(segment['seconds'] / PERIOD)
        began = time.monotonic()
        for i in range(1, frames + 1):
            due = began + i * segment['seconds'] / frames
            tour.wait_seconds(max(0, due - time.monotonic()))
            if time.monotonic() - due > .2:
                raise TimeoutError('Timing slipped; refusing to skip ahead')
            point = interpolate(segment['start'], segment['end'], i / frames)
            send_frame(port, point)
            if i % 5 == 0 or i == frames:
                actual = current_pose(port, limits)
                lag = max(abs(actual[ch] - point[ch]) for ch in tour.CHANNELS)
                max_lag = max(max_lag, lag)
                if lag > 200:  # 50 µs pulse tracking error, not physical feedback.
                    raise ValueError(f'Pulse tracking lag exceeded 50 µs ({lag / 4:g})')
        # No artificial dwell. Only gate unsafe handoffs on actual pulse arrival.
        wait_arrival(port, segment['end'], limits)
        if segment['pose_id'] is not None:
            print(f"Reached saved pose {segment['pose_id']} — continuing without dwell", flush=True)
    final = wait_arrival(port, tour.PARK, limits)
    print('DONE — parked pulse µs:', {ch: value / 4 for ch, value in final.items()}, flush=True)
    print(f'Elapsed {time.monotonic() - started:.1f}s; peak pulse tracking lag {max_lag / 4:g} µs', flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run', action='store_true')
    args = parser.parse_args()
    limits = {cfg['ch']: cfg for cfg in helper._servos().values()}
    plan = build_plan()
    verify_plan(plan, limits)
    if not args.run:
        print(json.dumps(dict(dwell_seconds=0, duration_seconds=sum(s['seconds'] for s in plan),
                             segments=plan), indent=2))
        return
    tour.check_stop()
    if not single_instance.acquire():
        raise RuntimeError('Rex owns the servos; did not connect')
    try:
        time.sleep(1.5)
        import serial
        with serial.Serial(helper._maestro_port(), 9600, timeout=.2, exclusive=True) as port:
            try:
                run(port, plan, limits)
            except BaseException:
                print('ABORT — holding current pulses', flush=True)
                helper._hold_throttle(port, limits)
                raise
    finally:
        single_instance.release()


if __name__ == '__main__':
    main()
