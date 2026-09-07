"""Mac-side approach decisions. Firmware receives only ordinary drive setpoints."""
from dataclasses import dataclass
import math
from statistics import median


@dataclass
class Decision:
    lin: float = 0.
    ang: float = 0.
    result: str | None = None
    reason: str = ''


def target_in_frame(snapshot, person_id, track_id):
    visible = [p for p in snapshot.get('people', [])
               if not p.get('face_missing') and p.get('face_visible') is not False
               and p.get('face_box')]
    if person_id is not None:
        same = [p for p in visible if p.get('person_db_id') == person_id]
        if len(same) == 1:
            return same[0]
    same = [p for p in visible if track_id is not None and p.get('id') == track_id
            and (person_id is None or p.get('person_db_id') in (None, person_id))]
    return same[0] if len(same) == 1 else None


def face_range_m(person, frame_width, half_fov_deg=25., face_width_m=.16):
    """Approximate camera range, not an identification or a collision guarantee."""
    try:
        width = float(person['face_box'][2])
        fraction = width / float(frame_width)
        value = float(face_width_m) / (2 * math.tan(math.radians(half_fov_deg)) * fraction)
        return value if width > 0 and .2 <= value <= 10. else None
    except (KeyError, TypeError, ValueError, ZeroDivisionError):
        return None


class Approach:
    def __init__(self, now, stop_at, speed, accel=.35):
        self.started = now
        self.last_seen = now
        self.stop_at = stop_at
        self.speed = speed
        self.accel = max(.05, accel)
        self.near_since = None
        self.near_sample = 0
        self.sample_count = 0
        self.sample_stamp = None
        self.block_since = None
        self.ranges = []
        self.last_range = None
        self.last_xy = None
        self.travel = 0.
        self.clear_since = None
        self.reason = 'starting'

    def step(self, now, telemetry, target_range, bearing_deg, target_stamp=None):
        odom = telemetry.get('odom') or {}
        xy = (float(odom.get('x', 0)), float(odom.get('y', 0)))
        if self.last_xy is not None:
            self.travel += math.hypot(xy[0]-self.last_xy[0], xy[1]-self.last_xy[1])
        self.last_xy = xy
        if now-self.started >= 20 or self.travel >= 4:
            return Decision(result='aborted', reason='approach time/travel limit')
        received = telemetry.get('rx_monotonic')
        if received is None or now-received > .6:
            return Decision(result='aborted', reason='motion telemetry stale')
        tof = telemetry.get('tof_mm') or {}
        fronts = [float(tof[k])*.001 for k in ('fl', 'fr') if tof.get(k, -1) > 0]
        if not fronts:
            return Decision(result='aborted', reason='front clearance unavailable')
        # Continue observing the caller while danger avoidance holds the base.
        # Otherwise a brief obstacle erases fresh camera evidence and adds a
        # second, artificial camera-loss pause after clearance returns.
        stamp = now if target_stamp is None else target_stamp
        if target_range is not None and stamp != self.sample_stamp:
            self.sample_stamp = stamp
            self.sample_count += 1
            self.last_seen = now
            self.ranges = (self.ranges + [target_range])[-3:]
            self.last_range = median(self.ranges)
        front = min(fronts)
        blocked = (telemetry.get('state') == 'blocked'
                   and telemetry.get('blocked_dir') in ('front', 'both')) or front <= .2
        if blocked:
            self.near_since = None
            self.clear_since = None
            if self.block_since is None: self.block_since = now
            return Decision(result='blocked' if now-self.block_since >= 6 else None,
                            reason='front obstacle; holding caller')
        self.block_since = None
        if now-self.last_seen >= 8:
            return Decision(result='aborted', reason='caller lost')
        if now-self.last_seen > 1.2 or self.last_range is None:
            self.near_since = None
            return Decision(reason='waiting for caller camera track')
        remaining = self.last_range-self.stop_at-.08
        if remaining <= 0:
            if target_range is None:
                self.near_since = None
                return Decision(reason='confirming caller at stand-off')
            if self.near_since is None:
                self.near_since, self.near_sample = now, self.sample_count
            if (now-self.near_since >= .6 and self.sample_count-self.near_sample >= 2
                    and abs(float(odom.get('lin', 0))) <= .08):
                return Decision(result='completed', reason='caller at camera stand-off')
            return Decision(reason='settling at caller')
        self.near_since = None
        lin = min(self.speed, max(.04, math.sqrt(2*self.accel*max(0, remaining))))
        # The ESP32 adds avoidance. Do not counter-steer against it; restore the
        # observed caller bearing only after a clear corridor has persisted.
        clear = front >= 1. and all(tof.get(k, -1) >= 400 for k in ('lf','lb','rf','rb'))
        if not clear: self.clear_since = None
        elif self.clear_since is None: self.clear_since = now
        ang = 0.
        if (self.clear_since is not None and now-self.clear_since >= .4
                and bearing_deg is not None):
            ang = max(-.25, min(.25, math.radians(bearing_deg)*1.2))
        return Decision(lin, ang, reason='approaching tracked caller')
