"""Slow upward throttle-arm body language, owned by one background worker.

Audio callbacks only suggest phrase-scale beats; all serial work happens here.
Explicit social introductions briefly select the measured level extension.
"""
import logging
import random
import threading
import time

import config
from hardware import servos
from hardware.throttle_motion import (
    CHANNELS, PARK, TUCK, clearance_box, cold_start_park_known, remember_park, validate_pose,
)

_log = logging.getLogger(__name__)


def pose(shoulder, elbow, wrist):
    return dict(zip(CHANNELS, (round(value * 4) for value in (shoulder, elbow, wrist))))


REST = pose(1050, 1930, 1280)
STARTUP = (TUCK, pose(1200, 2000, 950), pose(920, 2180, 1480), REST)
IDLE = (pose(1100, 1870, 1190), pose(980, 2020, 1400),
        pose(1020, 1960, 1230), pose(1120, 1910, 1340), REST)
SPEECH = (
    (pose(840, 2140, 1980), pose(1000, 1950, 950)),
    (pose(930, 2070, 850), pose(1080, 1880, 1870)),
    (pose(650, 2280, 2110), pose(940, 2040, 1070)),
    (pose(790, 2190, 1010), pose(1020, 1930, 2030)),
)


# Measured level extension; low/high anchors stay inside the established boxes.
INTRODUCTION = pose(544, 650.25, 1484.5)
LOW = pose(1636, 1550, 1500)
HIGH = pose(544, 2340.25, 1575.5)
COMMAND_POSES = {"down": LOW, "offer": INTRODUCTION, "high_five": HIGH, "rest": REST}
PRIDE_WRIST = 2254 * 4  # Downward curl within the raised/intermediate clearance box.


def expressive_pose(target, mood="neutral", intensity=0.0, pride=False,
                    introducing=False):
    """Blend complete poses, preserving phrase variation as mood fades."""
    if introducing:
        result = dict(INTRODUCTION)
    else:
        anchor = (LOW if mood in {"sad", "bored", "resigned", "sleepy"} else
                  HIGH if mood in {"excited", "giddy", "happy", "proud"} else None)
        weight = max(0.0, min(1.0, intensity)) if anchor else 0.0
        if anchor is LOW:
            # Ambient sadness is only 0.4; linear blending barely lowered the
            # shoulder and let upward speech poses overwhelm the expression.
            # Keep continuous decay to neutral, but make moderate sadness legible.
            weight = 1.0 - (1.0 - weight) ** 4
        result = {ch: round(target[ch] * (1 - weight) +
                            (anchor or target)[ch] * weight) for ch in CHANNELS}
    if pride and not introducing:
        result[10] = PRIDE_WRIST
    return result


def expression_state():
    from intelligence import body_mood, pride
    mood, intensity = body_mood.current_mood()
    controller = _controller
    if controller is not None and not controller.done.is_set():
        with controller.lock:
            if (controller.speech_expression is not None
                    and (controller.cadence.speaking
                         or time.monotonic() < controller.speech_expression_until)):
                mood, intensity = controller.speech_expression
    return mood, intensity, pride.is_active()


def request_pose(name):
    """Queue a ten-second pose hold on the existing worker; never start hardware."""
    controller = _controller
    if (name not in COMMAND_POSES or not config.THROTTLE_ARM_ENABLED
            or controller is None or not controller.thread.is_alive()
            or controller.done.is_set() or controller.stop_event.is_set()
            or controller.park_event.is_set() or not servos._automatic_motion_allowed()
            or servos._program_servo_updates_blocked()):
        return False
    with controller.lock:
        if controller.base_hold:
            return False
        controller.pose_request = (name, object())
        controller.pose_until = None
        controller.introduction_until = 0.0
    return True


def introduction():
    """Suggest a bounded greeting; never start or recover the hardware worker."""
    controller = _controller
    if controller is not None and not controller.done.is_set():
        with controller.lock:
            controller.introduction_until = time.monotonic() + 8.0


class SpeechCadence:
    """Use audible pauses as phrase proxies, not one target per amplitude sample.

    Cooldown survives TTS sentence boundaries. One expiring cue is retained, so
    buffered audio cannot build up a queue of stale gestures after speech ends.
    """
    def __init__(self):
        self.speaking = False
        self.next_beat = 0.0
        self.last_beat = None
        self.ended_at = None
        self.first_voice = self.last_voice = self.quiet_since = self.cue = None
        self.opening_sent = self.pause_sent = False

    def begin(self, now):
        if not self.speaking:
            self.first_voice = self.last_voice = self.quiet_since = self.cue = None
            self.opening_sent = self.pause_sent = False
        self.speaking = True
        self.ended_at = None

    def end(self, now):
        if self.speaking:
            self.ended_at = now
        self.speaking = False
        self.cue = None

    def level(self, level, now):
        if not self.speaking:
            return
        if level >= 0.08:
            if self.first_voice is None:
                self.first_voice = now
            self.last_voice = now
            self.quiet_since = None
            self.pause_sent = False
            if not self.opening_sent and now - self.first_voice >= 1.2:
                self.cue = now
                self.opening_sent = True
        elif self.first_voice is not None:
            if self.quiet_since is None:
                self.quiet_since = now
            if (not self.pause_sent and now - self.quiet_since >= config.THROTTLE_SPEECH_PAUSE_SECS
                    and self.quiet_since - self.first_voice >= 0.8):
                self.cue = now
                self.pause_sent = True

    def due(self, now):
        if not self.speaking or now < self.next_beat or self.first_voice is None:
            return False
        if self.cue is not None and now - self.cue <= 1.5:
            return True
        # An unusually long unbroken sentence still gets an occasional gesture.
        last = self.first_voice if self.last_beat is None else self.last_beat
        return (self.last_voice is not None and now - self.last_voice < 0.4
                and now - self.first_voice >= 1.2
                and now - last >= config.THROTTLE_SPEECH_FALLBACK_SECS)

    def consume(self, now, gap):
        self.cue = None
        self.last_beat = now
        self.next_beat = now + gap


def validate_repertoire():
    limits = {cfg['ch']: cfg for cfg in config.THROTTLE_SERVO_CHANNELS.values()}
    normal = IDLE + tuple(p for gesture in SPEECH for p in gesture)
    for p in (PARK,) + STARTUP + normal:
        validate_pose(p, limits)
    for start, end in zip((PARK,) + STARTUP, STARTUP):
        if not clearance_box(start, end):
            raise ValueError('Invalid throttle startup path')
    for start in normal + STARTUP:
        for end in normal + (TUCK,):
            if not clearance_box(start, end):
                raise ValueError('Invalid throttle background transition')
    # Blends are contained in these endpoint boxes. Verify every anchor/overlay
    # can return through REST, including an interrupted greeting at shutdown.
    expressions = (LOW, HIGH, INTRODUCTION) + tuple(
        expressive_pose(p, pride=True) for p in normal + (LOW, HIGH))
    for target in expressions:
        validate_pose(target, limits)
        if not (clearance_box(target, REST) and clearance_box(REST, target)):
            raise ValueError('Invalid throttle expression bridge')
    if not clearance_box(TUCK, PARK):
        raise ValueError('Invalid throttle park path')


class Controller:
    def __init__(self):
        self.stop_event = threading.Event()
        self.park_event = threading.Event()
        self.done = threading.Event()
        self.lock = threading.Lock()
        self.cadence = SpeechCadence()
        self.thread = threading.Thread(target=self.run, name='throttle-arm', daemon=True)
        self.connection = None
        self.parked = False
        self.fault = None
        self.last_idle = self.last_gesture = None
        self.introduction_until = 0.0
        self.pose_request = None
        self.pose_until = None
        self.last_expression = None
        self.speech_expression = None
        self.speech_expression_until = 0.0
        self.base_hold = False
        self.base_ready = threading.Event()
        self.base_seq = None
        self.base_sent_at = None

    def _hold(self):
        if self.connection is not None:
            try:
                servos.hold_throttle_pose(self.connection)
            except Exception as exc:
                _log.warning('Throttle could not hold current pulses: %s', exc)

    def _interrupted(self, parking):
        if self.stop_event.is_set():
            return True
        if not parking and not servos._automatic_motion_allowed():
            self.park_event.set()
        if not parking and self.park_event.is_set():
            return True
        if servos.throttle_motion_blocked(parking=parking):
            raise InterruptedError('Throttle yielded to servo override/latch')
        return False

    def move(self, target, kind, duration, *, parking=False, cold_start=False):
        if self._interrupted(parking):
            self._hold()
            return False
        # A forward reach and a lowered mood can require a bent-elbow bridge.
        current = servos.read_throttle_pose(self.connection)
        if any(current.values()) and not clearance_box(current, target):
            if not (clearance_box(current, REST) and clearance_box(REST, target)):
                raise ValueError('No verified throttle expression transition')
            if not self.move(REST, kind, duration, parking=parking):
                return False
        try:
            servos.move_throttle_pose(
                self.connection, target, speed_caps=getattr(config, f'THROTTLE_{kind}_SPEED'),
                accel_caps=getattr(config, f'THROTTLE_{kind}_ACCEL'), duration=duration,
                cancel=self.stop_event, cold_start=cold_start, parking=parking,
            )
        except InterruptedError:
            # The head can latch between our interruption check and the wire
            # write. A pending park still takes over instead of faulting here.
            if self._interrupted(parking):
                self._hold()
                return False
            raise
        deadline = time.monotonic() + 25.0
        while time.monotonic() < deadline:
            if self._interrupted(parking):
                self._hold()
                return False
            actual = servos.read_throttle_pose(self.connection)
            if all(abs(actual[ch] - target[ch]) <= 2 for ch in CHANNELS):
                return True
            self.stop_event.wait(0.1)
        raise TimeoutError('Throttle output did not reach its target')

    def _park(self, *, for_base=False):
        kind = 'RETRACT' if for_base else 'PARK'
        duration = config.THROTTLE_RETRACT_MOVE_SECS if for_base else config.THROTTLE_PARK_MOVE_SECS
        for target in (TUCK, PARK):
            if not self.move(target, kind, duration, parking=True):
                return
        if for_base and self.stop_event.wait(config.THROTTLE_RETRACT_SETTLE_SECS):
            return
        # Friction retains this position with outputs off; remember ONLY a
        # completed park. Every writer invalidates the marker before moving.
        remember_park()
        self.parked = True
        _log.info('Throttle arm parked at 2272 / 2496 / 512 us')

    def run(self):
        try:
            validate_repertoire()
            self.connection = servos.throttle_connection()
            current = servos.read_throttle_pose(self.connection)
            if not any(current.values()) and cold_start_park_known():
                if not self.move(PARK, 'PARK', config.THROTTLE_PARK_MOVE_SECS,
                                 parking=True, cold_start=True):
                    return
            elif any(abs(current[ch] - PARK[ch]) > 2 for ch in CHANNELS):
                raise ValueError('Throttle startup requires the verified park; no automatic reposition')
            if self.park_event.is_set():
                remember_park()
                self.parked = True
                return
            for target in STARTUP:
                if not self.move(target, 'STARTUP', config.THROTTLE_STARTUP_MOVE_SECS):
                    break
            next_idle = time.monotonic() + random.uniform(*config.THROTTLE_IDLE_DWELL_SECS)
            while not self.stop_event.is_set():
                if not servos._automatic_motion_allowed():
                    self.park_event.set()
                if self.park_event.is_set():
                    self._park()
                    return
                if servos._program_servo_updates_blocked():
                    raise InterruptedError('Throttle yielded to servo override/latch')
                with self.lock:
                    base_hold = self.base_hold
                if base_hold:
                    if not self.base_ready.is_set():
                        self._park(for_base=True)
                        if not self.parked:
                            return
                        self.base_ready.set()
                    from hardware import motion
                    telemetry = motion.telemetry() or {}
                    now = time.monotonic()
                    with self.lock:
                        # Only fresh, post-command idle telemetry can release the arm.
                        # No telemetry / lost link / rejected command keeps it tucked.
                        received = telemetry.get('rx_monotonic', 0)
                        if (self.base_seq is not None and self.base_sent_at is not None
                                and now - self.base_sent_at >= 0.5
                                and self.base_sent_at < received <= now
                                and now - received < 0.5
                                and telemetry.get('cmd_seq', -1) >= self.base_seq
                                and telemetry.get('state') == 'idle'
                                and telemetry.get('owner') == 'auto'):
                            self.base_hold = False
                            self.base_ready.clear()
                        base_hold = self.base_hold
                    if base_hold:
                        self.stop_event.wait(0.1)
                        continue
                    self.parked = False
                    for target in STARTUP:
                        if not self.move(target, 'STARTUP', config.THROTTLE_STARTUP_MOVE_SECS):
                            break
                    self.last_expression = None
                    continue
                now = time.monotonic()
                with self.lock:
                    requested = self.pose_request
                    pose_until = self.pose_until
                if requested is not None:
                    if pose_until is None:
                        reached = self.move(COMMAND_POSES[requested[0]], 'IDLE', 2.0)
                        with self.lock:
                            if self.pose_request == requested and reached:
                                self.pose_until = time.monotonic() + 10.0
                    elif now >= pose_until:
                        with self.lock:
                            if self.pose_request == requested:
                                self.pose_request = None
                        self.last_expression = None
                    self.stop_event.wait(0.1)
                    continue
                now = time.monotonic()
                with self.lock:
                    speaking = self.cadence.speaking
                    beat = self.cadence.due(now)
                    settle = (not speaking and self.cadence.ended_at is not None
                              and now - self.cadence.ended_at >= config.THROTTLE_SPEECH_SETTLE_SECS)
                    if beat:
                        self.cadence.consume(now, random.uniform(*config.THROTTLE_SPEECH_GAP_SECS))
                    if settle:
                        self.cadence.ended_at = None
                mood, intensity, pride = expression_state()
                with self.lock:
                    introducing = now < self.introduction_until
                expression = (mood, round(intensity, 1), pride, introducing)
                def shaped(target):
                    return expressive_pose(target, mood, intensity, pride, introducing)
                if expression != self.last_expression:
                    self.last_expression = expression
                    rest = shaped(REST)
                    _log.info('Throttle expression mood=%s intensity=%.2f pride=%s intro=%s target_us=%s',
                              mood, intensity, pride, introducing,
                              [rest[ch] / 4 for ch in CHANNELS])
                    self.move(rest, 'IDLE', random.uniform(*config.THROTTLE_IDLE_MOVE_SECS))
                    next_idle = time.monotonic() + random.uniform(*config.THROTTLE_IDLE_DWELL_SECS)
                elif beat and not introducing:
                    choices = [i for i in range(len(SPEECH)) if i != self.last_gesture]
                    self.last_gesture = random.choice(choices)
                    for target in SPEECH[self.last_gesture]:
                        with self.lock:
                            if not self.cadence.speaking:
                                break
                        if not self.move(shaped(target), 'SPEECH', random.uniform(*config.THROTTLE_SPEECH_MOVE_SECS)):
                            break
                    next_idle = time.monotonic() + random.uniform(*config.THROTTLE_IDLE_DWELL_SECS)
                elif not introducing and (settle or (not speaking and now >= next_idle)):
                    target = REST if settle else random.choice([p for p in IDLE if p != self.last_idle])
                    self.last_idle = target
                    self.move(shaped(target), 'IDLE', random.uniform(*config.THROTTLE_IDLE_MOVE_SECS))
                    next_idle = time.monotonic() + random.uniform(*config.THROTTLE_IDLE_DWELL_SECS)
                self.stop_event.wait(0.1)
        except Exception as exc:
            self.fault = str(exc)
            self._hold()
            _log.warning('Throttle animation stopped; no automatic restart: %s', exc)
        finally:
            if self.stop_event.is_set() and not self.parked:
                self._hold()
            self.done.set()


_controller = None
_lifecycle_lock = threading.Lock()


def start():
    """Startup/wake only; speech callbacks cannot start or recover the worker."""
    global _controller
    if not config.THROTTLE_ARM_ENABLED or not servos.SERVOS_ENABLED or not servos.connected():
        return False
    with _lifecycle_lock:
        if _controller is not None:
            if _controller.thread.is_alive():
                return True
            if _controller.fault:
                return False
        if servos._program_servo_updates_blocked() or not servos._automatic_motion_allowed():
            return False
        _controller = Controller()
        if servos.speech_motion_active():
            _controller.cadence.begin(time.monotonic())
        _controller.thread.start()
        return True


def request_park():
    """Start parking without delaying the head's independent shutdown glide."""
    controller = _controller
    if controller is not None:
        controller.park_event.set()


def park():
    """Request parking and join before serial teardown; the head may be latched."""
    request_park()
    controller = _controller
    if controller is None:
        return True
    if not controller.done.wait(30.0):
        controller.stop_event.set()
        controller.thread.join(timeout=2.0)
        _log.warning('Throttle park timed out; stopping rather than resuming gestures')
        return False
    return controller.parked


def stop():
    controller = _controller
    if controller is not None and controller.thread.is_alive():
        controller.stop_event.set()
        controller.thread.join(timeout=2.0)


def speech_start(frame=None):
    controller = _controller
    if controller is not None and not controller.done.is_set():
        with controller.lock:
            controller.cadence.begin(time.monotonic())
            # The already-resolved frame is also driving LEDs and the other arm.
            # Even neutral speech replaces a stale offended body mood for this turn.
            controller.speech_expression = (
                (str(frame.get('affect') or 'neutral'), float(frame.get('intensity', 0)))
                if frame is not None else None)
            controller.speech_expression_until = 0.0


def speech_stop():
    controller = _controller
    if controller is not None and not controller.done.is_set():
        with controller.lock:
            now = time.monotonic()
            controller.cadence.end(now)
            controller.speech_expression_until = now + config.THROTTLE_SPEECH_SETTLE_SECS


def speech_level(level):
    controller = _controller
    if controller is not None and not controller.done.is_set():
        with controller.lock:
            controller.cadence.level(level, time.monotonic())


def prepare_base_motion(timeout=30.0):
    """Reserve a parked arm before a base command. Fail closed if unavailable."""
    if not config.THROTTLE_ARM_ENABLED:
        return True
    controller = _controller
    if controller is None or controller.done.is_set() or controller.stop_event.is_set():
        return False
    with controller.lock:
        controller.base_hold = True
        controller.pose_request = None
        controller.pose_until = None
        controller.base_seq = None
        controller.base_sent_at = None
        controller.introduction_until = 0.0
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if controller.done.is_set() or controller.stop_event.is_set():
            return False
        if controller.base_ready.wait(0.1):
            try:
                actual = servos.read_throttle_pose(controller.connection)
                return (not servos.throttle_motion_blocked(parking=True)
                        and all(abs(actual[ch] - PARK[ch]) <= 2 for ch in CHANNELS))
            except Exception:
                return False
    return False


def base_motion_sent(seq):
    controller = _controller
    if controller is not None:
        with controller.lock:
            controller.base_seq = seq
            controller.base_sent_at = time.monotonic()
