"""Offline data model for manually demonstrated throttle sequences (no hardware IO)."""
import json
import math
import re
import time
import uuid
from datetime import datetime, timezone
from pathlib import Path

CHANNELS = (8, 9, 10)


def validate_pose(pose, limits):
    if set(pose) != set(CHANNELS):
        raise ValueError('All three throttle pulse readings are required')
    for ch, value in pose.items():
        if type(value) is not int or not limits[ch]['min'] <= value <= limits[ch]['max']:
            raise ValueError(f'Channel {ch} is off, unreadable, or outside current limits')


class Recorder:
    def __init__(self, name, start_pose, limits, clock=time.monotonic):
        validate_pose(start_pose, limits)
        if not name.strip():
            raise ValueError('Enter a sequence name')
        self.clock = clock
        self.started = clock()
        self.data = dict(version=1, name=name.strip()[:120],
                         created_utc=datetime.now(timezone.utc).isoformat(),
                         units='quarter_microseconds', start_pose=dict(start_pose), events=[])

    def record(self, channel, target, speed, acceleration):
        if channel in CHANNELS:
            self.data['events'].append(dict(at=round(self.clock() - self.started, 6),
                                           channel=channel, target=target,
                                           speed=speed, acceleration=acceleration))

    def save(self, end_pose, limits, directory):
        validate_pose(end_pose, limits)
        if not self.data['events']:
            raise ValueError('No throttle movement recorded yet')
        duration = self.clock() - self.started
        if not 0 < duration <= 3600:
            raise ValueError('Recordings must be shorter than one hour')
        data = dict(self.data, end_pose=dict(end_pose), duration=duration)
        directory = Path(directory)
        directory.mkdir(parents=True, exist_ok=True)
        slug = re.sub(r'[^a-z0-9]+', '-', data['name'].lower()).strip('-')[:60] or 'sequence'
        path = directory / f'{slug}-{uuid.uuid4().hex[:8]}.json'
        with path.open('x', encoding='utf-8') as stream:
            json.dump(data, stream, indent=2)
            stream.write('\n')
        return path


def load_sequence(path, limits):
    if Path(path).stat().st_size > 5_000_000:
        raise ValueError('Sequence file is too large')
    data = json.loads(Path(path).read_text())
    if data.get('version') != 1 or data.get('units') != 'quarter_microseconds':
        raise ValueError('Unsupported sequence format')
    for key in ('start_pose', 'end_pose'):
        data[key] = {int(ch): value for ch, value in data[key].items()}
        validate_pose(data[key], limits)
    duration = data['duration']
    if not isinstance(duration, (int, float)) or not math.isfinite(duration) or not 0 < duration <= 3600:
        raise ValueError('Sequence duration must be between zero and one hour')
    events = data['events']
    if not isinstance(events, list) or not 0 < len(events) <= 100000:
        raise ValueError('Invalid sequence events')
    previous = 0
    for event in events:
        at = event['at']
        if not isinstance(at, (float, int)) or not math.isfinite(at) or not previous <= at <= duration:
            raise ValueError('Invalid event timing')
        ch = event['channel']
        if type(ch) is not int or ch not in CHANNELS:
            raise ValueError('Sequence can only move throttle channels')
        for key, lo, hi in (('target', limits[ch]['min'], limits[ch]['max']),
                            ('speed', 1, 16383), ('acceleration', 1, 255)):
            if type(event[key]) is not int or not lo <= event[key] <= hi:
                raise ValueError(f'Invalid {key} for channel {ch}')
        previous = at
    return data


def require_start_pose(data, current, limits):
    validate_pose(current, limits)
    if any(abs(current[ch] - data['start_pose'][ch]) > 4 for ch in CHANNELS):
        required = ' / '.join(f"{data['start_pose'][ch] / 4:g}" for ch in CHANNELS)
        raise ValueError(f'Manually return to start S / E / W: {required} µs before playback')
