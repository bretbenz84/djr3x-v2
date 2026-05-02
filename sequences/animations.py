"""
sequences/animations.py — Choreographed servo + LED sequences for DJ-R3X.

Each function coordinates hardware/servos.py, hardware/leds_head.py, and
hardware/leds_chest.py into a named, timed behavior.  All sequences are
synchronous; callers run them in threads as needed.

Background behaviors run as daemon threads: servos.breathing_thread() for the
sinusoidal headlift oscillation, and animations.wander_thread() for slow
multi-channel idle head movements.  This module owns all triggered sequences.

Emotion → LED pattern reference (encoded in chest Arduino firmware):
    neutral  → RandomBlocks2, normal brightness
    excited  → AllRed, full brightness (255)
    sad      → AllBlue, dim (55)
    angry    → rapid red strobe (255)
    happy    → confetti, normal brightness
"""

import random
import threading
import time
import logging

import config
import state as _state_module
from state import State as _State
from hardware import servos, leds_head, leds_chest
from world_state import world_state

_log = logging.getLogger(__name__)

# Set while a TTS utterance is in progress — gates both speaking gestures and wander.
_speaking = threading.Event()
_motion_lock = threading.Lock()
_arm_motion_lock = threading.Lock()
_body_beat_lock = threading.Lock()
_last_directed_look: str | None = None


def speech_activity_start() -> None:
    """Mark TTS as active so wander loops stand down during playback."""
    _speaking.set()


def speech_activity_stop() -> None:
    """Clear the TTS activity gate used by wander loops."""
    _speaking.clear()

# ---------------------------------------------------------------------------
# Servo position constants (Pololu quarter-microseconds)
# ---------------------------------------------------------------------------

# Ch 0 — Neck: 1984–9984, neutral 6000, right = higher
NECK_CENTER   = 6000
NECK_LEFT     = 4000
NECK_RIGHT    = 8000
NECK_FAR_LEFT = 2500
NECK_FAR_RIGHT = 9500

# Ch 1 — Headlift: 1984–7744, neutral 6000, higher = up (larger value = head physically higher)
HEADLIFT_FLOOR   = 1984  # servo minimum — head fully lowered, shutdown/startup rest pose
HEADLIFT_DROOP   = 3000  # head drooped low (sleep)
HEADLIFT_DOWN    = 4800  # head below neutral (sad, lowered)
HEADLIFT_NEUTRAL = 6000
HEADLIFT_UP      = 7000  # head above neutral (excited, happy, nod)
HEADLIFT_HIGH    = 7500  # head raised high (surprised)

# Ch 2 — Headtilt: 3904–5504, neutral 4320, INVERTED (low value = head tilted up)
HEADTILT_NEUTRAL     = 4320
HEADTILT_UP          = 3904
HEADTILT_DOWN        = 5504
HEADTILT_SLIGHT_UP   = 4000
HEADTILT_SLIGHT_DOWN = 4700

# Ch 3 — Visor: 4544–6976, neutral 6000, higher = more open
VISOR_CLOSED  = 4544   # sleep / privacy — covers camera lens
VISOR_HALF    = 6400   # default resting open — clear of camera lens
VISOR_NEUTRAL = 6000
VISOR_OPEN    = 6976   # max — required before any camera capture

# Ch 4 — Elbow: 6300–7560, neutral 6720
ELBOW_NEUTRAL = 6720
ELBOW_UP      = 6300
ELBOW_DOWN    = 7560

# Ch 5 — Hand: 1984–9984, neutral 6000
HAND_NEUTRAL = 6000
HAND_LEFT    = 3500
HAND_RIGHT   = 8500

# Ch 6 — Pokerarm (left decorative): 3968–8000, neutral 6000
POKERARM_NEUTRAL = 6000
POKERARM_OUT     = 4500
POKERARM_IN      = 7500

# Ch 7 — Heroarm (right arm pivot): 3968–8000, neutral 6000
HEROARM_NEUTRAL = 6000
HEROARM_FORWARD = 4800
HEROARM_BACK    = 7200


# ---------------------------------------------------------------------------
# Body beats
# ---------------------------------------------------------------------------

_BODY_BEAT_HEAD_CHANNELS = (0, 1, 2, 3)
_BODY_BEAT_ARM_CHANNELS = (4, 5, 6, 7)
_BODY_BEAT_CHANNELS: dict[str, tuple[int, ...]] = {
    "suspicious_glance": _BODY_BEAT_HEAD_CHANNELS,
    "proud_dj_pose": _BODY_BEAT_HEAD_CHANNELS + _BODY_BEAT_ARM_CHANNELS,
    "offended_recoil": _BODY_BEAT_HEAD_CHANNELS + (4, 5, 7),
    "thinking_tilt": _BODY_BEAT_HEAD_CHANNELS,
    "dramatic_visor_peek": _BODY_BEAT_HEAD_CHANNELS,
    "tiny_victory_dance": _BODY_BEAT_HEAD_CHANNELS + _BODY_BEAT_ARM_CHANNELS,
}
_BODY_BEAT_ARM_NAMES = {
    name
    for name, channels in _BODY_BEAT_CHANNELS.items()
    if any(channel in _BODY_BEAT_ARM_CHANNELS for channel in channels)
}


def _channel_name(channel: int) -> str | None:
    for name, cfg in config.SERVO_CHANNELS.items():
        if int(cfg["ch"]) == channel:
            return name
    return None


def _channel_neutral(channel: int) -> int:
    name = _channel_name(channel)
    if not name:
        return 6000
    return int(config.SERVO_CHANNELS[name]["neutral"])


def _current_body_pose(channels: tuple[int, ...]) -> dict[int, int]:
    try:
        positions = (world_state.get("self_state") or {}).get("servo_positions") or {}
    except Exception:
        positions = {}

    pose: dict[int, int] = {}
    for channel in channels:
        name = _channel_name(channel)
        default = _channel_neutral(channel)
        try:
            pose[channel] = int(positions.get(name, default)) if name else default
        except (TypeError, ValueError):
            pose[channel] = default
    return pose


def _body_beat_allowed() -> bool:
    try:
        return _state_module.get_state() not in (_State.SLEEP, _State.SHUTDOWN)
    except Exception:
        return True


def _move_body(targets: dict[int, int], *, step_us: int = 70, step_delay: float = 0.01) -> None:
    servos.move_to(targets, step_us=step_us, step_delay=step_delay)


def _restore_body_pose(snapshot: dict[int, int], *, step_us: int = 55, step_delay: float = 0.012) -> None:
    if snapshot:
        servos.move_to(snapshot, step_us=step_us, step_delay=step_delay)


def _beat_suspicious_glance(snapshot: dict[int, int]) -> None:
    side = random.choice([-1, 1])
    _move_body(
        {
            0: NECK_CENTER + side * 1250,
            1: HEADLIFT_NEUTRAL + 220,
            2: HEADTILT_SLIGHT_DOWN,
            3: VISOR_NEUTRAL,
        },
        step_us=80,
        step_delay=0.008,
    )
    time.sleep(0.16)
    _move_body({0: NECK_CENTER - side * 420, 3: VISOR_HALF}, step_us=90, step_delay=0.007)
    time.sleep(0.10)
    _restore_body_pose(snapshot)


def _beat_proud_dj_pose(snapshot: dict[int, int]) -> None:
    _move_body(
        {
            1: HEADLIFT_HIGH,
            2: HEADTILT_SLIGHT_UP,
            3: VISOR_OPEN,
            4: ELBOW_UP,
            5: HAND_RIGHT,
            6: POKERARM_OUT,
            7: HEROARM_FORWARD,
        },
        step_us=85,
        step_delay=0.008,
    )
    time.sleep(0.28)
    _move_body({0: NECK_CENTER + 500, 5: HAND_LEFT}, step_us=105, step_delay=0.006)
    time.sleep(0.11)
    _move_body({0: NECK_CENTER - 500, 5: HAND_RIGHT}, step_us=105, step_delay=0.006)
    time.sleep(0.11)
    _restore_body_pose(snapshot)


def _beat_offended_recoil(snapshot: dict[int, int]) -> None:
    side = random.choice([-1, 1])
    # Headtilt is inverted: lower values tilt Rex upward, so this is the
    # chin-up "excuse me?" recoil.
    _move_body(
        {
            0: NECK_CENTER + side * 520,
            1: HEADLIFT_HIGH,
            2: HEADTILT_UP,
            3: VISOR_OPEN,
            4: ELBOW_UP,
            5: HAND_NEUTRAL,
            7: HEROARM_BACK,
        },
        step_us=120,
        step_delay=0.006,
    )
    time.sleep(0.18)
    _move_body({0: NECK_CENTER - side * 900, 5: HAND_LEFT if side > 0 else HAND_RIGHT}, step_us=100, step_delay=0.006)
    time.sleep(0.18)
    _restore_body_pose(snapshot)


def _beat_thinking_tilt(snapshot: dict[int, int]) -> None:
    side = random.choice([-1, 1])
    _move_body(
        {
            0: NECK_CENTER + side * 850,
            1: HEADLIFT_NEUTRAL + 260,
            2: HEADTILT_SLIGHT_UP,
            3: VISOR_HALF,
        },
        step_us=55,
        step_delay=0.014,
    )
    time.sleep(0.36)
    _move_body({0: NECK_CENTER + side * 1150, 2: HEADTILT_NEUTRAL}, step_us=45, step_delay=0.014)
    time.sleep(0.16)
    _restore_body_pose(snapshot)


def _beat_dramatic_visor_peek(snapshot: dict[int, int]) -> None:
    side = random.choice([-1, 1])
    _move_body({1: HEADLIFT_UP, 2: HEADTILT_SLIGHT_UP, 3: VISOR_CLOSED}, step_us=115, step_delay=0.006)
    time.sleep(0.10)
    _move_body({3: VISOR_OPEN, 0: NECK_CENTER + side * 700}, step_us=130, step_delay=0.005)
    time.sleep(0.20)
    _restore_body_pose(snapshot, step_us=65, step_delay=0.010)


def _beat_tiny_victory_dance(snapshot: dict[int, int]) -> None:
    _move_body(
        {
            1: HEADLIFT_UP,
            2: HEADTILT_SLIGHT_UP,
            3: VISOR_OPEN,
            4: ELBOW_UP,
            5: HAND_RIGHT,
            6: POKERARM_OUT,
            7: HEROARM_FORWARD,
        },
        step_us=90,
        step_delay=0.006,
    )
    for side in (-1, 1, -1, 1):
        _move_body(
            {
                0: NECK_CENTER + side * 520,
                1: HEADLIFT_UP if side > 0 else HEADLIFT_NEUTRAL + 320,
                5: HAND_RIGHT if side > 0 else HAND_LEFT,
            },
            step_us=120,
            step_delay=0.004,
        )
        time.sleep(0.06)
    _move_body({1: HEADLIFT_HIGH, 4: ELBOW_DOWN, 6: POKERARM_IN}, step_us=110, step_delay=0.005)
    time.sleep(0.08)
    _restore_body_pose(snapshot)


_BODY_BEAT_RUNNERS = {
    "suspicious_glance": _beat_suspicious_glance,
    "proud_dj_pose": _beat_proud_dj_pose,
    "offended_recoil": _beat_offended_recoil,
    "thinking_tilt": _beat_thinking_tilt,
    "dramatic_visor_peek": _beat_dramatic_visor_peek,
    "tiny_victory_dance": _beat_tiny_victory_dance,
}

_BODY_BEAT_ALIASES = {
    "suspicious": "suspicious_glance",
    "side_eye": "suspicious_glance",
    "wrong_answer": "suspicious_glance",
    "game_wrong": "suspicious_glance",
    "proud": "proud_dj_pose",
    "dj_start": "proud_dj_pose",
    "dj_pose": "proud_dj_pose",
    "offended": "offended_recoil",
    "insult": "offended_recoil",
    "insult_recoil": "offended_recoil",
    "thinking": "thinking_tilt",
    "think": "thinking_tilt",
    "daily_double": "dramatic_visor_peek",
    "visor_peek": "dramatic_visor_peek",
    "dj_stop": "dramatic_visor_peek",
    "correct_answer": "tiny_victory_dance",
    "game_correct": "tiny_victory_dance",
    "victory": "tiny_victory_dance",
}


def _canonical_body_beat(name: str) -> str | None:
    normalized = "_".join(str(name or "").strip().lower().replace("-", " ").split())
    canonical = _BODY_BEAT_ALIASES.get(normalized, normalized)
    return canonical if canonical in _BODY_BEAT_RUNNERS else None


def body_beat_names() -> list[str]:
    """Return the named physical beats callers can trigger semantically."""
    return sorted(_BODY_BEAT_RUNNERS)


def play_body_beat(name: str, *, async_: bool = True) -> bool:
    """
    Play a named physical punctuation beat.

    The beat runs in a daemon thread by default so conversation/game/DJ logic can
    keep moving while Rex performs a short embodied reaction.
    """
    canonical = _canonical_body_beat(name)
    if not canonical:
        _log.debug("[animations] unknown body beat: %r", name)
        return False

    if async_:
        threading.Thread(
            target=_run_body_beat,
            args=(canonical,),
            daemon=True,
            name=f"body_beat_{canonical}",
        ).start()
        return True
    return _run_body_beat(canonical)


def _run_body_beat(name: str) -> bool:
    if not _body_beat_allowed():
        return False
    if not _body_beat_lock.acquire(blocking=False):
        return False

    uses_arm = name in _BODY_BEAT_ARM_NAMES
    arm_acquired = False
    try:
        if uses_arm:
            servos.pause_arm_idle()
            arm_acquired = _arm_motion_lock.acquire(blocking=False)
            if not arm_acquired:
                return False
        snapshot = _current_body_pose(_BODY_BEAT_CHANNELS[name])
        with _motion_lock:
            _BODY_BEAT_RUNNERS[name](snapshot)
        return True
    except Exception as exc:
        _log.debug("[animations] body beat %s failed: %s", name, exc)
        return False
    finally:
        if arm_acquired:
            _arm_motion_lock.release()
        if uses_arm:
            servos.resume_arm_idle()
        _body_beat_lock.release()

# ---------------------------------------------------------------------------
# Startup / shutdown
# ---------------------------------------------------------------------------

def startup() -> None:
    """Power-on: chest startup burst, head raises and looks around in parallel."""
    leds_chest.startup()
    leds_head.active()
    leds_head.set_eye_color(255, 200, 0)    # warm gold boot-up eyes

    # Raise head + open visor in a background thread while the main thread
    # runs the neck sweep — gives the impression of waking up and looking around
    # simultaneously, instead of head-up-then-look.
    lift_thread = threading.Thread(
        target=servos.move_to,
        args=({1: HEADLIFT_NEUTRAL, 2: HEADTILT_NEUTRAL, 3: VISOR_HALF},),
        kwargs={"step_us": 25, "step_delay": 0.025},
        daemon=True,
        name="startup_lift",
    )
    lift_thread.start()

    # Look around as if waking up — randomly choose left-right or right-left.
    if random.random() < 0.5:
        servos.move_to({0: NECK_LEFT},  step_us=40, step_delay=0.025)
        time.sleep(0.3)
        servos.move_to({0: NECK_RIGHT}, step_us=40, step_delay=0.025)
        time.sleep(0.3)
    else:
        servos.move_to({0: NECK_RIGHT}, step_us=40, step_delay=0.025)
        time.sleep(0.3)
        servos.move_to({0: NECK_LEFT},  step_us=40, step_delay=0.025)
        time.sleep(0.3)

    # Wait for the head lift to finish before centering the neck.
    lift_thread.join()

    # Return to center.
    servos.move_to({0: NECK_CENTER}, step_us=40, step_delay=0.025)
    time.sleep(0.2)


def shutdown() -> None:
    """Shutdown: stop breathing, slowly close visor, droop head to rest position, LEDs off."""
    servos.stop_breathing()
    time.sleep(0.1)   # let breathing thread exit before we move headlift

    # Close visor first, then slowly center neck and droop head to the rest/startup pose.
    servos.move_to({3: VISOR_CLOSED}, step_us=30, step_delay=0.025)
    time.sleep(0.3)
    servos.move_to(
        {0: NECK_CENTER, 1: HEADLIFT_FLOOR, 2: HEADTILT_DOWN},
        step_us=25, step_delay=0.025,
    )
    time.sleep(0.5)
    leds_head.off()
    leds_chest.off()

# ---------------------------------------------------------------------------
# Sleep / wake
# ---------------------------------------------------------------------------

def sleep() -> None:
    """Sleep: visor closes (covers camera), head droops, dim red chest breath, eyes off."""
    leds_chest.sleep()
    leds_head.sleep()
    servos.set_servo(3, VISOR_CLOSED)       # close visor before head droops
    time.sleep(0.4)
    servos.set_servos({
        0: NECK_CENTER,
        1: HEADLIFT_DROOP,
        2: HEADTILT_DOWN,
    })


def wake() -> None:
    """Wake from sleep: head raises, visor opens, active LEDs restore."""
    leds_chest.active()
    servos.set_servos({
        1: HEADLIFT_NEUTRAL,
        2: HEADTILT_NEUTRAL,
        3: VISOR_HALF,
    })
    time.sleep(0.3)
    leds_head.active()
    leds_head.set_eye_color(255, 200, 0)


# ---------------------------------------------------------------------------
# Idle
# ---------------------------------------------------------------------------

def idle() -> None:
    """Idle state: normal brightness LEDs, servos smoothly to neutral."""
    leds_chest.idle()
    leds_head.idle()
    servos.neutral(step_us=30)


def wander_thread() -> None:
    """
    Background thread: slow, subtle multi-channel head movements during IDLE/ACTIVE.
    Randomly picks from neck scans, headtilt shifts, thoughtful glances, and resets.
    Suppressed while speaking or in SLEEP/SHUTDOWN states.
    Call as a daemon thread from main.py alongside breathing_thread.

    When nobody is detected in frame, the thread runs more frequently and uses
    wider neck/headtilt sweeps — Rex actively looks around as if scanning for
    company, instead of holding gaze on a person who isn't there.
    """
    while True:
        # No one in frame → wander more often and sweep further.
        alone = not world_state.get("people")

        if alone:
            time.sleep(random.uniform(2.0, 5.0))
        else:
            time.sleep(random.uniform(4.0, 10.0))

        cur = _state_module.get_state()
        if cur not in (_State.IDLE, _State.ACTIVE):
            continue
        if _speaking.is_set():
            continue

        # Wider sweeps when alone — head turns farther and tilts more.
        neck_scan_range = (700, 1800) if alone else (300, 700)
        neck_lean_range = (300, 1000) if alone else (200, 500)
        neck_nudge_range = (400, 1000) if alone else (-250, 250)
        tilt_scan_amp    = 180 if alone else 80

        choice = random.randint(0, 4)

        if choice == 0:
            # Slow neck turn with slight headtilt — like scanning the room
            side = random.choice([-1, 1])
            neck = NECK_CENTER + side * random.randint(*neck_scan_range)
            tilt = HEADTILT_NEUTRAL + random.randint(-tilt_scan_amp, tilt_scan_amp)
            servos.move_to({0: neck, 2: tilt}, step_us=20, step_delay=0.03)
            time.sleep(random.uniform(0.8, 2.0))
            servos.move_to({0: NECK_CENTER, 2: HEADTILT_NEUTRAL}, step_us=20, step_delay=0.03)

        elif choice == 1:
            # Thoughtful upward glance — head lifts and tilts slightly up
            lift = HEADLIFT_NEUTRAL + random.randint(100, 250)
            servos.move_to({1: lift, 2: HEADTILT_SLIGHT_UP}, step_us=20, step_delay=0.03)
            time.sleep(random.uniform(0.8, 1.5))
            servos.move_to({1: HEADLIFT_NEUTRAL, 2: HEADTILT_NEUTRAL}, step_us=20, step_delay=0.03)

        elif choice == 2:
            # Downward contemplative look — slight neck lean + head lower + tilt down
            side = random.choice([-1, 1])
            neck = NECK_CENTER + side * random.randint(*neck_lean_range)
            servos.move_to(
                {0: neck, 1: HEADLIFT_NEUTRAL - 200, 2: HEADTILT_SLIGHT_DOWN},
                step_us=20, step_delay=0.03,
            )
            time.sleep(random.uniform(1.0, 2.0))
            servos.move_to({0: NECK_CENTER, 1: HEADLIFT_NEUTRAL, 2: HEADTILT_NEUTRAL}, step_us=20, step_delay=0.03)

        elif choice == 3:
            # Slow re-center — settle back to neutral from any drift
            servos.move_to(
                {0: NECK_CENTER, 1: HEADLIFT_NEUTRAL, 2: HEADTILT_NEUTRAL},
                step_us=15, step_delay=0.03,
            )

        else:
            # Subtle visor adjustment + small neck lean
            visor_nudge = VISOR_HALF + random.randint(-80, 80)
            if alone:
                neck_nudge = NECK_CENTER + random.choice([-1, 1]) * random.randint(*neck_nudge_range)
            else:
                neck_nudge = NECK_CENTER + random.randint(*neck_nudge_range)
            servos.move_to({3: visor_nudge, 0: neck_nudge}, step_us=20, step_delay=0.03)
            time.sleep(random.uniform(0.8, 1.5))
            servos.move_to({3: VISOR_HALF, 0: NECK_CENTER}, step_us=20, step_delay=0.03)


def arm_wander_thread() -> None:
    """
    Background thread: heroarm and pokerarm pick a random target at either 50%
    or 100% of their range from neutral, in a random direction, during
    IDLE/ACTIVE. Independent from the head wander so arm and head motion don't
    synchronise. Suppressed while speaking or in SLEEP/SHUTDOWN. Call as a
    daemon thread from main.py.
    """
    # Both arms have ~2000 qus of travel on each side of their 6000 neutral.
    ARM_HALF_RANGE = 2000

    while True:
        time.sleep(random.uniform(6.0, 15.0))

        cur = _state_module.get_state()
        if cur not in (_State.IDLE, _State.ACTIVE):
            continue
        if _speaking.is_set() or servos.arm_idle_paused():
            continue

        targets: dict[int, int] = {}
        for ch, neutral in ((7, HEROARM_NEUTRAL), (6, POKERARM_NEUTRAL)):
            magnitude = random.choice([0.5, 1.0]) * ARM_HALF_RANGE
            direction = random.choice([-1, 1])
            targets[ch] = neutral + int(direction * magnitude)

        # Slow, smooth interpolation. step_delay=0.20 → ~100 qus/sec, so a 50%
        # sweep takes ~10 s and a full sweep ~20 s — visible but unhurried.
        if not _arm_motion_lock.acquire(blocking=False):
            continue
        try:
            if not _speaking.is_set() and not servos.arm_idle_paused():
                servos.move_to(targets, step_us=20, step_delay=0.20)
        finally:
            _arm_motion_lock.release()


# ---------------------------------------------------------------------------
# Speech
# ---------------------------------------------------------------------------

def _speaking_loop() -> None:
    """Background: subtle expressive head movements during a TTS utterance."""
    while _speaking.is_set():
        choice = random.randint(0, 3)
        if choice == 0:
            # Slight neck turn — shift gaze as if addressing the room
            side = random.choice([-1, 1])
            target = {0: NECK_CENTER + side * random.randint(250, 500)}
        elif choice == 1:
            # Emphasis lift — head rises slightly on an important phrase
            target = {1: HEADLIFT_NEUTRAL + random.randint(80, 200)}
        elif choice == 2:
            # Expressive head tilt
            tilt = random.choice([-1, 1]) * random.randint(50, 120)
            target = {2: HEADTILT_NEUTRAL + tilt}
        else:
            # Drift back toward neutral — natural reset between gestures
            target = {0: NECK_CENTER, 1: HEADLIFT_NEUTRAL, 2: HEADTILT_NEUTRAL}

        servos.move_to(target, step_us=50, step_delay=0.02)

        hold = random.uniform(1.0, 3.0)
        deadline = time.monotonic() + hold
        while _speaking.is_set() and time.monotonic() < deadline:
            time.sleep(0.1)


def speech_start(emotion: str = "neutral") -> None:
    """
    Call at the start of a TTS utterance.
    Sends the emotion pattern to both Arduinos, adjusts head pose to match,
    and starts a background thread for subtle expressive head movements.
    """
    _speaking.set()
    threading.Thread(target=_speaking_loop, daemon=True, name="speech_gestures").start()

    leds_chest.speak(emotion)
    leds_head.speak(emotion)
    leds_head.set_eye_emotion(emotion)
    servos.set_breathing_emotion(emotion)

    if emotion == "excited":
        servos.set_servos({3: VISOR_OPEN, 1: HEADLIFT_UP, 2: HEADTILT_SLIGHT_UP})
    elif emotion == "sad":
        servos.set_servos({3: VISOR_HALF, 1: HEADLIFT_DOWN, 2: HEADTILT_SLIGHT_DOWN})
    elif emotion == "angry":
        servos.set_servos({3: VISOR_HALF, 1: HEADLIFT_NEUTRAL})
    elif emotion == "happy":
        servos.set_servos({3: VISOR_OPEN, 1: HEADLIFT_UP})
    else:
        servos.set_servos({3: VISOR_HALF, 1: HEADLIFT_NEUTRAL})


def speech_stop() -> None:
    """Call when TTS finishes. Stops gesture thread and resets head pose to idle."""
    _speaking.clear()
    leds_head.speak_stop()
    leds_chest.idle()
    servos.set_breathing_emotion("neutral")
    servos.set_servos({0: NECK_CENTER, 3: VISOR_HALF, 1: HEADLIFT_NEUTRAL, 2: HEADTILT_NEUTRAL})


def speech_level(amplitude: int) -> None:
    """Drive mouth LED brightness from audio buffer level (0–255)."""
    leds_head.speak_level(amplitude)


# ---------------------------------------------------------------------------
# Head expressions
# ---------------------------------------------------------------------------

def nod(count: int = 2) -> None:
    """Acknowledgment nod — headlift up/down cycle."""
    for _ in range(count):
        servos.set_servo(1, HEADLIFT_UP)
        time.sleep(0.12)
        servos.set_servo(1, HEADLIFT_NEUTRAL)
        time.sleep(0.12)


def headshake(count: int = 2) -> None:
    """Disagreement — neck left/right sweep, returns to center."""
    for _ in range(count):
        servos.set_servo(0, NECK_CENTER - 1800)
        time.sleep(0.14)
        servos.set_servo(0, NECK_CENTER + 1800)
        time.sleep(0.14)
    servos.set_servo(0, NECK_CENTER)


def visor_flutter(count: int = 2) -> None:
    """Expressive punctuation — quick open/half cycle."""
    for _ in range(count):
        servos.set_servo(3, VISOR_OPEN)
        time.sleep(0.10)
        servos.set_servo(3, VISOR_HALF)
        time.sleep(0.10)


def thinking() -> None:
    """Rex considering something: slight upward tilt, sideways glance."""
    servos.set_servos({2: HEADTILT_SLIGHT_UP, 0: NECK_CENTER + 1000})


def surprised() -> None:
    """Genuine surprise beat: quick head-up + visor fully open."""
    servos.set_servos({1: HEADLIFT_HIGH, 3: VISOR_OPEN})
    time.sleep(0.3)


# ---------------------------------------------------------------------------
# Gaze / neck tracking
# ---------------------------------------------------------------------------

def look_left(amount: int = 2000) -> None:
    servos.set_servo(0, max(1984, NECK_CENTER - amount))


def look_right(amount: int = 2000) -> None:
    servos.set_servo(0, min(9984, NECK_CENTER + amount))


def look_center() -> None:
    servos.set_servo(0, NECK_CENTER)


def camera_pose() -> None:
    """Visor fully open + neck centered before image capture. Waits 0.5 s to settle."""
    servos.set_servos({0: NECK_CENTER, 3: VISOR_OPEN})
    time.sleep(0.5)


def _world_self_state() -> dict:
    try:
        return world_state.get("self_state")
    except Exception:
        return {}


def _current_lateral_direction() -> str | None:
    try:
        pos = servos.get_servo(0)
    except Exception:
        pos = None
    if pos is None:
        pos = (_world_self_state().get("servo_positions") or {}).get("neck")
    try:
        neck = int(pos)
    except (TypeError, ValueError):
        return None
    if neck <= NECK_CENTER - 400:
        return "left"
    if neck >= NECK_CENTER + 400:
        return "right"
    return None


def _opposite_direction() -> str:
    lateral = _current_lateral_direction()
    if lateral == "left":
        return "right"
    if lateral == "right":
        return "left"
    if _last_directed_look == "left":
        return "right"
    if _last_directed_look == "right":
        return "left"
    if _last_directed_look == "up":
        return "down"
    if _last_directed_look == "down":
        return "up"
    return "right"


def _record_directed_look(direction: str, target: str = "") -> None:
    global _last_directed_look
    _last_directed_look = direction
    try:
        self_state = world_state.get("self_state")
        self_state["last_directed_look"] = direction
        self_state["last_directed_look_at"] = time.time()
        self_state["last_look_target"] = target or None
        world_state.update("self_state", self_state)
    except Exception:
        pass


def directed_look_pose(direction: str = "current", target: str = "") -> str:
    """
    Move Rex's head toward a requested direction for a user-directed visual check.

    Returns the normalized direction actually used. Unlike camera_pose(), this
    intentionally preserves side/up/down gaze so the next frame represents what
    Rex was asked to inspect.
    """
    norm = (direction or "current").strip().lower()
    if norm in {"here", "this", "that", "there", "pointed", "show"}:
        norm = "current"
    elif norm in {"other", "other_way", "opposite", "opposite_way"}:
        norm = _opposite_direction()
    elif norm in {"centre", "front", "forward", "ahead", "straight"}:
        norm = "center"
    elif norm not in {"left", "right", "up", "down", "center", "current"}:
        norm = "current"

    neck_offset = int(getattr(config, "DIRECTED_LOOK_NECK_OFFSET_QUS", 2200))
    lift_offset = int(getattr(config, "DIRECTED_LOOK_HEADLIFT_OFFSET_QUS", 900))
    tilt_offset = int(getattr(config, "DIRECTED_LOOK_HEADTILT_OFFSET_QUS", 450))
    settle = float(getattr(config, "DIRECTED_LOOK_SETTLE_SECS", 0.65))
    step_us = int(getattr(config, "DIRECTED_LOOK_STEP_QUS", 30))
    step_delay = float(getattr(config, "DIRECTED_LOOK_STEP_DELAY_SECS", 0.032))

    targets = {3: VISOR_OPEN}
    if norm == "left":
        targets[0] = max(1984, NECK_CENTER - neck_offset)
    elif norm == "right":
        targets[0] = min(9984, NECK_CENTER + neck_offset)
    elif norm == "up":
        targets[1] = min(7744, HEADLIFT_NEUTRAL + lift_offset)
        # Headtilt is inverted: lower values tilt the head/camera upward.
        targets[2] = max(3904, HEADTILT_NEUTRAL - tilt_offset)
    elif norm == "down":
        targets[1] = max(1984, HEADLIFT_NEUTRAL - lift_offset)
        targets[2] = min(5504, HEADTILT_NEUTRAL + tilt_offset)
    elif norm == "center":
        targets.update({0: NECK_CENTER, 1: HEADLIFT_NEUTRAL, 2: HEADTILT_NEUTRAL})

    with _motion_lock:
        servos.move_to(targets, step_us=step_us, step_delay=step_delay)
        time.sleep(settle)
    _record_directed_look(norm, target)
    return norm


# ---------------------------------------------------------------------------
# Arm
# ---------------------------------------------------------------------------

def arm_hero_pose() -> None:
    """Heroarm forward, elbow up, hand neutral — confident presentation pose."""
    servos.set_servos({7: HEROARM_FORWARD, 4: ELBOW_UP, 5: HAND_NEUTRAL})


def arm_idle() -> None:
    """Return right arm assembly to neutral."""
    servos.set_servos({4: ELBOW_NEUTRAL, 5: HAND_NEUTRAL, 7: HEROARM_NEUTRAL})


def arm_fidget() -> None:
    """Small randomized hand nudge — idle micro-behavior."""
    nudge = random.randint(-400, 400)
    servos.set_servo(5, HAND_NEUTRAL + nudge)
    time.sleep(0.5)
    servos.set_servo(5, HAND_NEUTRAL)


def arm_rhythm_tick(beat_phase: float) -> None:
    """
    Subtle elbow dip locked to music beat phase (0.0–1.0 per beat).
    Call from the DJ playback loop on each detected beat downbeat.
    """
    if beat_phase < 0.15:
        servos.set_servo(4, ELBOW_DOWN)
    elif beat_phase < 0.5:
        servos.set_servo(4, ELBOW_NEUTRAL)


def arm_wave(count: int | None = None) -> None:
    """Wave the right arm by raising/lowering the elbow a few times."""
    if count is None:
        count = int(getattr(config, "WAVE_COUNT", 3))
    count = max(1, min(6, int(count)))
    hold = float(getattr(config, "WAVE_HOLD_SECS", 0.12))
    step_us = int(getattr(config, "WAVE_STEP_QUS", 55))
    step_delay = float(getattr(config, "WAVE_STEP_DELAY_SECS", 0.012))

    servos.pause_arm_idle()
    try:
        with _arm_motion_lock:
            with _motion_lock:
                servos.move_to(
                    {7: HEROARM_FORWARD, 5: HAND_NEUTRAL, 4: ELBOW_NEUTRAL},
                    step_us=step_us,
                    step_delay=step_delay,
                )
                for _ in range(count):
                    servos.move_to({4: ELBOW_UP}, step_us=step_us, step_delay=step_delay)
                    time.sleep(hold)
                    servos.move_to({4: ELBOW_DOWN}, step_us=step_us, step_delay=step_delay)
                    time.sleep(hold)
                servos.move_to(
                    {4: ELBOW_NEUTRAL, 5: HAND_NEUTRAL, 7: HEROARM_NEUTRAL},
                    step_us=step_us,
                    step_delay=step_delay,
                )
    finally:
        servos.resume_arm_idle()


# ---------------------------------------------------------------------------
# Composite reactions
# ---------------------------------------------------------------------------

def excited_burst() -> None:
    """Full excited reaction: arm up, head bob, visor open, chest AllRed."""
    leds_chest.speak("excited")
    leds_head.speak("excited")
    leds_head.set_eye_emotion("excited")
    servos.set_servos({3: VISOR_OPEN, 1: HEADLIFT_UP, 7: HEROARM_FORWARD, 4: ELBOW_UP})
    time.sleep(0.25)
    servos.set_servo(1, HEADLIFT_NEUTRAL)
    time.sleep(0.15)
    servos.set_servo(1, HEADLIFT_UP)
    time.sleep(0.15)
    servos.set_servo(1, HEADLIFT_NEUTRAL)


def roast_pose() -> None:
    """Lean into a roast — slight head tilt down + sideways look."""
    servos.set_servos({2: HEADTILT_SLIGHT_DOWN, 3: VISOR_HALF, 0: NECK_CENTER + 600})


def dismissal() -> None:
    """Dismissive head-turn away."""
    servos.set_servos({0: NECK_CENTER + 2500, 2: HEADTILT_SLIGHT_DOWN})


def return_to_neutral() -> None:
    """Smoothly return all channels to neutral positions."""
    servos.neutral(step_us=30)
