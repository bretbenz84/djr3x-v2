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
from intelligence import emotion_orchestrator
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

# Ch 0 — Neck: configured range midpoint, right = higher
def _channel_midpoint(name: str) -> int:
    cfg = config.SERVO_CHANNELS[name]
    return (int(cfg["min"]) + int(cfg["max"])) // 2


NECK_CENTER   = _channel_midpoint("neck")
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
VISOR_SQUINT  = 5272   # ~halfway between neutral (6000) and fully-closed (4544): the
                       # visor drops DOWN over the eyes — a transient "displeased
                       # squint/glower" reacting to an insult. Below the lens-clear
                       # floor, so beats dip here briefly then return (NEVER sustained —
                       # the body_mood path stays clamped at the floor). Tune 5000–5700
                       # for a deeper/shallower glower.
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

# The pose shutdown() / sleep() park the head in — head fully lowered, looking down,
# visor closed, neck centred. startup() seeds its slow raise from this so it never
# depends on a fresh-connect proprioception read to know where it's starting from.
SHUTDOWN_REST_POSE = {
    0: NECK_CENTER,
    1: HEADLIFT_FLOOR,
    2: HEADTILT_DOWN,
    3: VISOR_CLOSED,
}

# Idle arm wander should read as intentional arm motion, not servo creep. The
# hero arm gets the broadest swing; pokerarm stays a quieter secondary accent.
_IDLE_ARM_WAIT_RANGE_SECS = (4.0, 9.0)
_IDLE_ARM_STEP_QUS = 70
_IDLE_POKERARM_STEP_QUS = 210
_IDLE_ARM_STEP_DELAY_SECS = 0.045
_IDLE_HEROARM_SWING_RANGE_QUS = (1300, 2000)
_IDLE_HEROARM_MIN_TRAVEL_QUS = 900
_IDLE_POKERARM_SWING_RANGE_QUS = (800, 1500)
_IDLE_POKERARM_MIN_TRAVEL_QUS = 550


# ---------------------------------------------------------------------------
# Body beats
# ---------------------------------------------------------------------------

_BODY_BEAT_HEAD_CHANNELS = (0, 1, 2, 3)
_BODY_BEAT_ARM_CHANNELS = (4, 5, 6, 7)
_BODY_BEAT_CHANNELS: dict[str, tuple[int, ...]] = {
    "agreement_nod": _BODY_BEAT_HEAD_CHANNELS,
    "anger_flash": _BODY_BEAT_HEAD_CHANNELS + (4, 5, 7),
    "disagreement_shake": _BODY_BEAT_HEAD_CHANNELS,
    "disbelief_stare": _BODY_BEAT_HEAD_CHANNELS,
    "disgust_recoil": _BODY_BEAT_HEAD_CHANNELS + (5, 7),
    "suspicious_glance": _BODY_BEAT_HEAD_CHANNELS,
    "giddy_wiggle": _BODY_BEAT_HEAD_CHANNELS + _BODY_BEAT_ARM_CHANNELS,
    "happy_bounce": _BODY_BEAT_HEAD_CHANNELS + (4, 7),
    "proud_dj_pose": _BODY_BEAT_HEAD_CHANNELS + _BODY_BEAT_ARM_CHANNELS,
    "offended_recoil": _BODY_BEAT_HEAD_CHANNELS + (4, 5, 7),
    "sad_droop": _BODY_BEAT_HEAD_CHANNELS,
    "surprise_pop": _BODY_BEAT_HEAD_CHANNELS,
    "thinking_tilt": _BODY_BEAT_HEAD_CHANNELS,
    "dramatic_visor_peek": _BODY_BEAT_HEAD_CHANNELS,
    "tiny_victory_dance": _BODY_BEAT_HEAD_CHANNELS + _BODY_BEAT_ARM_CHANNELS,
    "eye_roll": _BODY_BEAT_HEAD_CHANNELS,
    "double_take": _BODY_BEAT_HEAD_CHANNELS,
    "mic_drop": _BODY_BEAT_HEAD_CHANNELS + (7,),
    "spit_take": _BODY_BEAT_HEAD_CHANNELS,
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


def _clamp_channel_position(channel: int, position: int) -> int:
    name = _channel_name(channel)
    if not name:
        return int(position)
    cfg = config.SERVO_CHANNELS[name]
    return max(int(cfg["min"]), min(int(cfg["max"]), int(position)))


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


def _idle_arm_target(
    channel: int,
    neutral: int,
    current: int,
    swing_range_qus: tuple[int, int],
    min_travel_qus: int,
) -> int:
    """
    Pick a visible idle arm target.

    When the arm is already offset from neutral, bias the next move across the
    body so idle poses do not linger in tiny same-side corrections.
    """
    if current >= neutral + min_travel_qus:
        direction = -1
    elif current <= neutral - min_travel_qus:
        direction = 1
    else:
        direction = random.choice([-1, 1])

    magnitude = random.randint(*swing_range_qus)
    return _clamp_channel_position(channel, neutral + direction * magnitude)


def _idle_arm_wander_targets() -> dict[int, int]:
    current = _current_body_pose((7, 6))
    return {
        7: _idle_arm_target(
            7,
            HEROARM_NEUTRAL,
            current.get(7, HEROARM_NEUTRAL),
            _IDLE_HEROARM_SWING_RANGE_QUS,
            _IDLE_HEROARM_MIN_TRAVEL_QUS,
        ),
        6: _idle_arm_target(
            6,
            POKERARM_NEUTRAL,
            current.get(6, POKERARM_NEUTRAL),
            _IDLE_POKERARM_SWING_RANGE_QUS,
            _IDLE_POKERARM_MIN_TRAVEL_QUS,
        ),
    }


def _body_beat_allowed() -> bool:
    try:
        return _state_module.get_state() not in (_State.SLEEP, _State.SHUTDOWN)
    except Exception:
        return True


# Spontaneous (self-directed) comedic beats are rate-limited so Rex doesn't mug
# nonstop on his own. EXPLICIT requests ("do a mic drop") and deterministic
# event/mood/gamepad beats are NOT gated — only self-fires pass spontaneous=True.
_last_spontaneous_beat_at = 0.0
_spontaneous_beat_lock = threading.Lock()


def spontaneous_beat_allowed() -> bool:
    """True if enough time has elapsed to fire another SELF-DIRECTED comedic beat."""
    try:
        gap = float(getattr(config, "COMEDY_BEAT_MIN_GAP_SECS", 6.0))
    except Exception:
        gap = 6.0
    if gap <= 0:
        return True
    with _spontaneous_beat_lock:
        return (time.monotonic() - _last_spontaneous_beat_at) >= gap


def note_spontaneous_beat() -> None:
    """Record that a self-directed comedic beat just fired (starts the cooldown)."""
    global _last_spontaneous_beat_at
    with _spontaneous_beat_lock:
        _last_spontaneous_beat_at = time.monotonic()


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


def _beat_surprise_pop(snapshot: dict[int, int]) -> None:
    # Eyebrow equivalent: visor snaps fully open while the head pops up.
    # The headtilt stays parked: the ~5 lb head pivots on an 8 mm rod, and
    # snapping it (especially forward/down on recovery) is too violent — the
    # surprise reads from full visor + full headlift instead.
    _move_body(
        {
            0: NECK_CENTER,
            1: HEADLIFT_HIGH,
            3: VISOR_OPEN,
        },
        step_us=150,
        step_delay=0.004,
    )
    time.sleep(0.20)
    _move_body({1: HEADLIFT_UP}, step_us=90, step_delay=0.006)
    time.sleep(0.08)
    _restore_body_pose(snapshot)


def _beat_anger_flash(snapshot: dict[int, int]) -> None:
    side = random.choice([-1, 1])
    _move_body(
        {
            0: NECK_CENTER + side * 320,
            1: HEADLIFT_UP,
            2: HEADTILT_DOWN,
            3: VISOR_SQUINT,   # dip the visor below neutral — narrowed, displeased
                               # squint (this is the beat insults route to)
            4: ELBOW_UP,
            5: HAND_RIGHT if side > 0 else HAND_LEFT,
            7: HEROARM_FORWARD,
        },
        step_us=130,
        step_delay=0.005,
    )
    time.sleep(0.10)
    for turn in (-side, side, -side):
        _move_body({0: NECK_CENTER + turn * 480}, step_us=150, step_delay=0.004)
        time.sleep(0.045)
    _move_body({3: VISOR_HALF, 2: HEADTILT_SLIGHT_DOWN}, step_us=85, step_delay=0.006)
    time.sleep(0.08)
    _restore_body_pose(snapshot)


def _beat_disgust_recoil(snapshot: dict[int, int]) -> None:
    side = random.choice([-1, 1])
    away_hand = HAND_LEFT if side > 0 else HAND_RIGHT
    _move_body(
        {
            0: NECK_CENTER + side * 900,
            1: HEADLIFT_UP,
            2: HEADTILT_DOWN,
            3: VISOR_NEUTRAL,
            5: away_hand,
            7: HEROARM_BACK,
        },
        step_us=115,
        step_delay=0.006,
    )
    time.sleep(0.18)
    _move_body({0: NECK_CENTER + side * 1350, 3: VISOR_HALF}, step_us=100, step_delay=0.006)
    time.sleep(0.12)
    _restore_body_pose(snapshot)


def _beat_happy_bounce(snapshot: dict[int, int]) -> None:
    _move_body(
        {
            0: NECK_CENTER,
            1: HEADLIFT_UP,
            2: HEADTILT_SLIGHT_UP,
            3: VISOR_OPEN,
            4: ELBOW_UP,
            7: HEROARM_FORWARD,
        },
        step_us=95,
        step_delay=0.006,
    )
    for lift in (HEADLIFT_HIGH, HEADLIFT_NEUTRAL + 250, HEADLIFT_UP):
        _move_body({1: lift}, step_us=120, step_delay=0.004)
        time.sleep(0.055)
    _restore_body_pose(snapshot)


def _beat_giddy_wiggle(snapshot: dict[int, int]) -> None:
    # Headtilt stays parked (heavy head, 8 mm rod) — the giddiness is all
    # visor-wide, headlift bounce, and neck/arm wiggle.
    _move_body(
        {
            1: HEADLIFT_UP,
            3: VISOR_OPEN,
            4: ELBOW_UP,
            5: HAND_RIGHT,
            6: POKERARM_OUT,
            7: HEROARM_FORWARD,
        },
        step_us=115,
        step_delay=0.004,
    )
    for side in (-1, 1, -1, 1, -1):
        _move_body(
            {
                0: NECK_CENTER + side * 460,
                1: HEADLIFT_HIGH if side > 0 else HEADLIFT_UP,
                5: HAND_RIGHT if side > 0 else HAND_LEFT,
                6: POKERARM_OUT if side > 0 else POKERARM_IN,
            },
            step_us=145,
            step_delay=0.0035,
        )
        time.sleep(0.045)
    _restore_body_pose(snapshot)


def _beat_disbelief_stare(snapshot: dict[int, int]) -> None:
    side = random.choice([-1, 1])
    _move_body(
        {
            0: NECK_CENTER + side * 240,
            1: HEADLIFT_NEUTRAL + 220,
            2: HEADTILT_DOWN,
            3: VISOR_OPEN,
        },
        step_us=70,
        step_delay=0.010,
    )
    time.sleep(0.28)
    _move_body({0: NECK_CENTER - side * 240}, step_us=95, step_delay=0.006)
    time.sleep(0.08)
    _move_body({0: NECK_CENTER + side * 140}, step_us=95, step_delay=0.006)
    time.sleep(0.10)
    _restore_body_pose(snapshot)


def _beat_agreement_nod(snapshot: dict[int, int]) -> None:
    _move_body({3: VISOR_OPEN, 1: HEADLIFT_UP, 2: HEADTILT_SLIGHT_UP}, step_us=95, step_delay=0.005)
    time.sleep(0.05)
    for _ in range(2):
        _move_body({1: HEADLIFT_NEUTRAL + 160, 2: HEADTILT_SLIGHT_DOWN}, step_us=120, step_delay=0.004)
        time.sleep(0.055)
        _move_body({1: HEADLIFT_UP, 2: HEADTILT_SLIGHT_UP}, step_us=120, step_delay=0.004)
        time.sleep(0.055)
    _restore_body_pose(snapshot)


def _beat_disagreement_shake(snapshot: dict[int, int]) -> None:
    _move_body({1: HEADLIFT_NEUTRAL + 140, 2: HEADTILT_SLIGHT_DOWN, 3: VISOR_HALF}, step_us=85, step_delay=0.006)
    for side in (-1, 1, -1, 1):
        _move_body({0: NECK_CENTER + side * 1150}, step_us=145, step_delay=0.0035)
        time.sleep(0.050)
    _move_body({0: NECK_CENTER}, step_us=120, step_delay=0.004)
    time.sleep(0.04)
    _restore_body_pose(snapshot)


def _beat_sad_droop(snapshot: dict[int, int]) -> None:
    _move_body(
        {
            0: NECK_CENTER,
            1: HEADLIFT_DOWN,
            2: HEADTILT_SLIGHT_DOWN,
            3: VISOR_HALF,
        },
        step_us=45,
        step_delay=0.018,
    )
    time.sleep(0.38)
    _restore_body_pose(snapshot, step_us=45, step_delay=0.016)


def _beat_eye_roll(snapshot: dict[int, int]) -> None:
    # No movable eyes — the visor IS the brow. An eye-roll reads as a brow-lift
    # (visor pops open), the chin tipping up ("ugh, really"), and a slow neck arc
    # across and back. Languid on purpose: a slow roll, not a snap.
    side = random.choice([-1, 1])
    _move_body(
        {2: HEADTILT_UP, 3: VISOR_OPEN, 0: NECK_CENTER + side * 320},
        step_us=60,
        step_delay=0.013,
    )
    time.sleep(0.12)
    _move_body(
        {0: NECK_CENTER - side * 320, 2: HEADTILT_SLIGHT_DOWN},
        step_us=52,
        step_delay=0.015,
    )
    time.sleep(0.08)
    _restore_body_pose(snapshot)


def _beat_double_take(snapshot: dict[int, int]) -> None:
    # Glance away, casual — then SNAP back with a visor pop and a head lift: the
    # classic "wait... WHAT." The comedy is the speed difference between the lazy
    # look-away and the sharp return.
    side = random.choice([-1, 1])
    _move_body(
        {0: NECK_CENTER + side * 760, 3: VISOR_HALF},
        step_us=120,
        step_delay=0.005,
    )
    time.sleep(0.16)
    # The snap-back keeps the headtilt parked (too much mass to whip on the
    # tilt rod) — the "WHAT" is sold by the visor pop and the full head lift.
    _move_body(
        {0: NECK_CENTER, 3: VISOR_OPEN, 1: HEADLIFT_HIGH},
        step_us=230,
        step_delay=0.002,
    )
    time.sleep(0.13)
    _restore_body_pose(snapshot)


def _beat_mic_drop(snapshot: dict[int, int]) -> None:
    # Present the "mic" — hero arm forward, chin up, visor wide, supremely smug —
    # hold the beat, then DROP it: the arm swings back/down and the head turns away
    # dismissively. The punctuation on a line he knows landed.
    side = random.choice([-1, 1])
    _move_body(
        {7: HEROARM_FORWARD, 1: HEADLIFT_UP, 2: HEADTILT_UP, 3: VISOR_OPEN},
        step_us=110,
        step_delay=0.006,
    )
    time.sleep(0.24)
    _move_body(
        {7: HEROARM_BACK, 0: NECK_CENTER + side * 720, 2: HEADTILT_SLIGHT_DOWN},
        step_us=210,
        step_delay=0.003,
    )
    time.sleep(0.16)
    _restore_body_pose(snapshot)


def _beat_spit_take(snapshot: dict[int, int]) -> None:
    # Sharp backward recoil — head snaps up-and-back, visor pops wide (shock), a
    # quick neck flinch, then a small settle. The "you said WHAT" reaction: fast
    # and jerky where the eye-roll is slow.
    side = random.choice([-1, 1])
    # No headtilt in the recoil — this is the fastest move in the file, and the
    # heavy head can't take a full-range tilt whip. Lift + visor carry the shock.
    _move_body(
        {1: HEADLIFT_HIGH, 3: VISOR_OPEN, 0: NECK_CENTER + side * 360},
        step_us=255,
        step_delay=0.002,
    )
    time.sleep(0.10)
    _move_body(
        {0: NECK_CENTER - side * 240, 1: HEADLIFT_UP},
        step_us=170,
        step_delay=0.004,
    )
    time.sleep(0.06)
    _restore_body_pose(snapshot)


_BODY_BEAT_RUNNERS = {
    "agreement_nod": _beat_agreement_nod,
    "anger_flash": _beat_anger_flash,
    "disagreement_shake": _beat_disagreement_shake,
    "disbelief_stare": _beat_disbelief_stare,
    "disgust_recoil": _beat_disgust_recoil,
    "giddy_wiggle": _beat_giddy_wiggle,
    "happy_bounce": _beat_happy_bounce,
    "suspicious_glance": _beat_suspicious_glance,
    "proud_dj_pose": _beat_proud_dj_pose,
    "offended_recoil": _beat_offended_recoil,
    "sad_droop": _beat_sad_droop,
    "surprise_pop": _beat_surprise_pop,
    "thinking_tilt": _beat_thinking_tilt,
    "dramatic_visor_peek": _beat_dramatic_visor_peek,
    "tiny_victory_dance": _beat_tiny_victory_dance,
    "eye_roll": _beat_eye_roll,
    "double_take": _beat_double_take,
    "mic_drop": _beat_mic_drop,
    "spit_take": _beat_spit_take,
}

_BODY_BEAT_ALIASES = {
    "agree": "agreement_nod",
    "agreement": "agreement_nod",
    "yes": "agreement_nod",
    "yes_nod": "agreement_nod",
    "nod": "agreement_nod",
    "angry": "anger_flash",
    "anger": "anger_flash",
    "mad": "anger_flash",
    "furious": "anger_flash",
    "disagree": "disagreement_shake",
    "disagreement": "disagreement_shake",
    "no": "disagreement_shake",
    "nope": "disagreement_shake",
    "headshake": "disagreement_shake",
    "head_shake": "disagreement_shake",
    "disbelief": "disbelief_stare",
    "incredulous": "disbelief_stare",
    "skeptical_stare": "disbelief_stare",
    "disgust": "disgust_recoil",
    "disgusted": "disgust_recoil",
    "grossed_out": "disgust_recoil",
    "giddy": "giddy_wiggle",
    "giddy_joy": "giddy_wiggle",
    "joy": "giddy_wiggle",
    "glee": "giddy_wiggle",
    "happy": "happy_bounce",
    "happy_bounce": "happy_bounce",
    "sad": "sad_droop",
    "sadness": "sad_droop",
    "dejected": "sad_droop",
    "surprise": "surprise_pop",
    "surprised": "surprise_pop",
    "shocked": "surprise_pop",
    "startled": "surprise_pop",
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
    "eyeroll": "eye_roll",
    "roll_eyes": "eye_roll",
    "rolls_eyes": "eye_roll",
    "doubletake": "double_take",
    "micdrop": "mic_drop",
    "drop_the_mic": "mic_drop",
    "mic": "mic_drop",
    "spittake": "spit_take",
    "spit": "spit_take",
}


def _canonical_body_beat(name: str) -> str | None:
    normalized = "_".join(str(name or "").strip().lower().replace("-", " ").split())
    canonical = _BODY_BEAT_ALIASES.get(normalized, normalized)
    return canonical if canonical in _BODY_BEAT_RUNNERS else None


def body_beat_names() -> list[str]:
    """Return the named physical beats callers can trigger semantically."""
    return sorted(_BODY_BEAT_RUNNERS)


def play_body_beat(name: str, *, async_: bool = True, spontaneous: bool = False) -> bool:
    """
    Play a named physical punctuation beat.

    The beat runs in a daemon thread by default so conversation/game/DJ logic can
    keep moving while Rex performs a short embodied reaction.

    Set spontaneous=True for a SELF-DIRECTED beat (Rex deciding to perform on his
    own, with no explicit request or triggering event): these are frequency-gated
    by COMEDY_BEAT_MIN_GAP_SECS so he doesn't mug nonstop. Explicit requests and
    deterministic event/mood/gamepad beats leave it False and are never throttled.
    """
    canonical = _canonical_body_beat(name)
    if not canonical:
        _log.debug("[animations] unknown body beat: %r", name)
        return False

    if spontaneous and not spontaneous_beat_allowed():
        _log.debug("[animations] spontaneous body beat %r suppressed by frequency cooldown", canonical)
        return False
    if spontaneous:
        note_spontaneous_beat()

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

    # Servo-whir accent under the physical beat (audio/sound_effects owns the
    # cooldown, so back-to-back beats don't chirp every time). Best-effort.
    try:
        from audio import sound_effects
        sound_effects.play("servo")
    except Exception:
        pass

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
    #
    # Seed the sweep from the known shutdown rest pose rather than a fresh-connect
    # proprioception read: that first read is the least reliable, and if it comes
    # back wrong the floor->neutral interpolation collapses and the head jerks
    # straight up instead of rising slowly. We parked it at the rest pose on the
    # last shutdown()/sleep(), so we know exactly where it is.
    #
    # Pace: the old 25/0.025 (~1000 qus/s) wake read as sluggish (owner 2026-08-11:
    # "the startup servo movements are slow"). 60/0.02 streams ~3000 qus/s — still
    # under the startup profile's physical cap (speed 40 ≈ 4000 qus/s) and ramped
    # by the per-channel acceleration, so the fragile headtilt still GLIDES (never
    # snap the tilt — 5 lb head on an 8 mm rod), just ~3x brisker.
    lift_thread = threading.Thread(
        target=servos.move_to,
        args=({1: HEADLIFT_NEUTRAL, 2: HEADTILT_NEUTRAL, 3: VISOR_HALF},),
        kwargs={
            "step_us": 60,
            "step_delay": 0.02,
            "start": {ch: SHUTDOWN_REST_POSE[ch] for ch in (1, 2, 3)},
        },
        daemon=True,
        name="startup_lift",
    )
    lift_thread.start()

    # Look around as if waking up — randomly choose left-right or right-left. The
    # first turn starts from the centred rest pose for the same reason as the lift.
    # 70/0.025 (~2800 qus/s) keeps the look-around in step with the faster rise.
    neck_start = {0: SHUTDOWN_REST_POSE[0]}
    if random.random() < 0.5:
        servos.move_to({0: NECK_LEFT},  step_us=70, step_delay=0.025, start=neck_start)
        time.sleep(0.3)
        servos.move_to({0: NECK_RIGHT}, step_us=70, step_delay=0.025)
        time.sleep(0.3)
    else:
        servos.move_to({0: NECK_RIGHT}, step_us=70, step_delay=0.025, start=neck_start)
        time.sleep(0.3)
        servos.move_to({0: NECK_LEFT},  step_us=70, step_delay=0.025)
        time.sleep(0.3)

    # Wait for the head lift to finish before centering the neck.
    lift_thread.join()

    # Return to center.
    servos.move_to({0: NECK_CENTER}, step_us=70, step_delay=0.025)
    time.sleep(0.2)


# Boot-time "looking around the room while models load" motion. Tuned to read as
# unhurried, curious scanning — not the frantic full-swing of a search.
_BOOT_SCAN_STEP_US = 90
_BOOT_SCAN_STEP_DELAY_SECS = 0.018
_BOOT_SCAN_DWELL_RANGE_SECS = (0.25, 0.6)
_BOOT_SCAN_MIN_NECK_JUMP_FRACTION = 0.22   # ensure each look visibly moves the head
_BOOT_SCAN_DOWN_LIFT_FRACTION = 0.32       # how far the headlift may dip below neutral
_BOOT_SCAN_DOWN_TILT_FRACTION = 0.50       # how far the headtilt may angle down


def boot_scan_thread(stop_event: "threading.Event") -> None:
    """Look around the room while startup models load, so the head isn't frozen.

    Sweeps to randomized two-axis poses (neck across its full configured range,
    paired with a varied downward pitch) until ``stop_event`` is set, then recenters
    to neutral so consciousness / face tracking inherit a known pose. ``move_to``
    clamps every target to the servo limits, so the full-range neck sweep stays safe.
    Runs as a daemon thread during the preload window only — nothing else drives the
    head until consciousness starts well afterward.
    """
    try:
        neck_cfg = config.SERVO_CHANNELS["neck"]
        lift_cfg = config.SERVO_CHANNELS["headlift"]
        tilt_cfg = config.SERVO_CHANNELS["headtilt"]
        neck_min, neck_max = int(neck_cfg["min"]), int(neck_cfg["max"])
        lift_neutral, tilt_neutral = int(lift_cfg["neutral"]), int(tilt_cfg["neutral"])
        # Pitch dips from level toward "down" (people sit/stand below the head camera).
        lift_down = lift_neutral - int((lift_neutral - int(lift_cfg["min"])) * _BOOT_SCAN_DOWN_LIFT_FRACTION)
        tilt_down = tilt_neutral + int((int(tilt_cfg["max"]) - tilt_neutral) * _BOOT_SCAN_DOWN_TILT_FRACTION)
        min_jump = int((neck_max - neck_min) * _BOOT_SCAN_MIN_NECK_JUMP_FRACTION)

        last_neck: int | None = None
        while not stop_event.is_set():
            if _speaking.is_set():
                # The boot line is talking — its speech gestures own the head; stand
                # down (like wander/breathing do) and resume scanning once it finishes.
                last_neck = None
                stop_event.wait(0.15)
                continue
            # Pick a neck target that's a clear jump from the last, so the head moves.
            neck = random.randint(neck_min, neck_max)
            for _ in range(4):
                if last_neck is None or abs(neck - last_neck) >= min_jump:
                    break
                neck = random.randint(neck_min, neck_max)
            last_neck = neck
            # Pair the turn with a random pitch so each move is two-axis (diagonal).
            f = random.random()
            lift = lift_neutral + int((lift_down - lift_neutral) * f)
            tilt = tilt_neutral + int((tilt_down - tilt_neutral) * f)
            servos.move_to(
                {0: neck, 1: lift, 2: tilt},
                step_us=_BOOT_SCAN_STEP_US,
                step_delay=_BOOT_SCAN_STEP_DELAY_SECS,
            )
            stop_event.wait(random.uniform(*_BOOT_SCAN_DWELL_RANGE_SECS))

        servos.move_to(
            {0: NECK_CENTER, 1: HEADLIFT_NEUTRAL, 2: HEADTILT_NEUTRAL},
            step_us=40,
            step_delay=0.02,
        )
    except Exception as exc:
        _log.debug("boot scan thread error: %s", exc)


def shutdown() -> None:
    """Shutdown: stop breathing, droop to the rest pose, LEDs off.

    Visor close, neck recenter, head-lift droop and head-tilt down all travel
    together in a SINGLE move_to so the droid powers down in one motion, instead
    of the old visor→tilt→lift sequence. (Headtilt is inverted: HEADTILT_DOWN is
    the "looking down" value.)

    Timing: step_us=50 / step_delay=0.012 makes the droop brisk and apparent (~1s
    over the ~4000-unit head-lift travel) rather than the old ~4s crawl, while
    staying smooth (finer steps than the expressive gestures, which use 80-150).

    Motion profile: move_to only streams Set-Target steps; it never programs the
    Maestro's per-channel speed/accel. So the droop inherits whatever profile the
    last subsystem left on the head channels — and that is often a SLOW one
    (listening = speed 22 / accel 6, adaptive-rest = 35 / 6). At step_us=50 /
    step_delay=0.012 the software finishes streaming the FLOOR target in ~0.7s
    while the slow-capped servo is still mid-travel, then serial closes and the
    head is stranded near where it was (≈ neutral). We reset to a brisk profile
    here so the physical servo can actually keep up and reach the rest pose.
    """
    # The GUI's manual servo override freezes EVERY programmatic move — including
    # this one. If the operator shuts down with the override still on (field bug
    # 2026-07-16: droop silently no-oped, program exited with the head wherever
    # it was), the power-down pose must still win: the program is exiting, so
    # there is nothing left for the override to protect. Clear it first.
    try:
        servos.set_manual_override_enabled(False)
    except Exception:
        pass
    # Shutting down while asleep: the sleep latch would freeze the droop too.
    try:
        servos.release_sleep_latch()
    except Exception:
        pass

    servos.stop_breathing()
    time.sleep(0.1)   # let breathing thread exit before we move headlift

    # Clear any stale slow speed/accel left on the head channels before the droop.
    servos.set_motion_profile(
        config.HEAD_CHANNELS,
        speed=int(getattr(config, "SHUTDOWN_DROOP_SERVO_SPEED", 70)),
        acceleration=int(getattr(config, "SHUTDOWN_DROOP_SERVO_ACCELERATION", 14)),
    )

    servos.move_to(
        {3: VISOR_CLOSED, 0: NECK_CENTER, 1: HEADLIFT_FLOOR, 2: HEADTILT_DOWN},
        step_us=50, step_delay=0.012,
    )
    # Freeze the pose the INSTANT the rest targets are commanded — before the
    # settle sleep, not after. The Maestro finishes the physical travel on its
    # own; every quarter-second the latch is delayed is a window where another
    # thread (a late frame grab, a finishing audio clip's end_speech_motion)
    # can re-open the visor (field bug 2026-08-02: the shutdown clip ended
    # 4s before the old post-settle latch and drove the visor to neutral).
    try:
        servos.latch_shutdown_pose()
    except Exception:
        pass
    # Give the head time to physically arrive at FLOOR before LEDs off / serial
    # close. Don't rely on the shutdown-audio join window (skipped when audio is
    # disabled), or a correct-speed droop could still be cut short.
    time.sleep(float(getattr(config, "SHUTDOWN_DROOP_SETTLE_SECS", 0.8)))
    # Fade the LEDs out (lifelike power-down) rather than snapping them off. _shutdown()
    # already kicked off this fade in lockstep with the audio + droop; FADEOFF is
    # idempotent in firmware, so this is a harmless re-assert (and keeps shutdown()
    # correct if it's ever called on its own).
    leds_head.fade_off()
    leds_chest.fade_off()

# ---------------------------------------------------------------------------
# Sleep / wake
# ---------------------------------------------------------------------------

def sleep() -> None:
    """Sleep: return to the shutdown/rest pose without tearing hardware down.

    Carries the same protections shutdown() earned in the field (see its
    docstring): reset the head/arm motion profile so a stale slow speed cap
    can't strand the glide short of the pose, keep the visor target asserted
    through the final move so a racing writer can't leave it half-open, and
    latch the pose the moment it's commanded — wake() (or shutdown(), when Rex
    is powered off in his sleep) releases the latch. Field 2026-08-13: the
    sleep ack clip's end_speech_motion re-opened the visor around the glide
    and Rex "slept" with the visor visibly open.
    """
    leds_chest.sleep()
    leds_head.sleep()
    servos.pause_arm_idle()
    servos.set_motion_profile(
        list(config.HEAD_CHANNELS) + list(config.ARM_CHANNELS),
        speed=int(getattr(config, "SHUTDOWN_DROOP_SERVO_SPEED", 70)),
        acceleration=int(getattr(config, "SHUTDOWN_DROOP_SERVO_ACCELERATION", 14)),
    )
    servos.move_to({3: VISOR_CLOSED}, step_us=25, step_delay=0.035)
    time.sleep(0.25)
    servos.move_to(
        {
            0: NECK_CENTER,
            1: HEADLIFT_FLOOR,
            2: HEADTILT_DOWN,
            3: VISOR_CLOSED,
            4: ELBOW_NEUTRAL,
            5: HAND_NEUTRAL,
            6: POKERARM_NEUTRAL,
            7: HEROARM_NEUTRAL,
        },
        step_us=25,
        step_delay=0.035,
    )
    servos.latch_sleep_pose()


def wake() -> None:
    """Wake from sleep: head raises, visor opens, active LEDs restore."""
    servos.release_sleep_latch()
    leds_chest.active()
    servos.move_to(
        {
            1: HEADLIFT_NEUTRAL,
            2: HEADTILT_NEUTRAL,
            3: VISOR_HALF,
        },
        step_us=35,
        step_delay=0.02,
    )
    leds_head.active()
    leds_head.set_eye_color(255, 200, 0)
    servos.resume_arm_idle()


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
        if _face_tracking_holding_gaze():
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
    Background thread: heroarm and pokerarm pick visible offset poses during
    IDLE/ACTIVE. Independent from the head wander so arm and head motion don't
    synchronise. Suppressed while speaking or in SLEEP/SHUTDOWN. Call as a
    daemon thread from main.py.
    """
    while True:
        time.sleep(random.uniform(*_IDLE_ARM_WAIT_RANGE_SECS))

        cur = _state_module.get_state()
        if cur not in (_State.IDLE, _State.ACTIVE):
            continue
        if _speaking.is_set() or servos.arm_idle_paused():
            continue

        if not _arm_motion_lock.acquire(blocking=False):
            continue
        try:
            if not _speaking.is_set() and not servos.arm_idle_paused():
                servos.move_to(
                    _idle_arm_wander_targets(),
                    step_us={
                        7: _IDLE_ARM_STEP_QUS,
                        6: _IDLE_POKERARM_STEP_QUS,
                    },
                    step_delay=_IDLE_ARM_STEP_DELAY_SECS,
                )
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

    frame = emotion_orchestrator.frame_for_speech(emotion)
    led_emotion = frame.led_style
    emotion_orchestrator.publish_frame(frame)

    leds_chest.speak(led_emotion)
    leds_head.speak(led_emotion)
    leds_head.set_eye_emotion(led_emotion)
    servos.set_breathing_emotion(led_emotion)

    if frame.affect in {"excited", "giddy", "surprised"}:
        # No headtilt command here: set_servos is an instant target jump, and the
        # heavy head on its 8 mm tilt rod can't take that snap. Excitement and
        # surprise read from the visor at max + headlift at max instead.
        servos.set_servos({3: VISOR_OPEN, 1: HEADLIFT_HIGH})
    elif frame.affect in {"sad", "sleepy"}:
        servos.set_servos({3: VISOR_HALF, 1: HEADLIFT_DOWN, 2: HEADTILT_SLIGHT_DOWN})
    elif frame.affect == "angry":
        servos.set_servos({3: VISOR_HALF, 1: HEADLIFT_NEUTRAL})
    elif frame.affect == "disgusted":
        servos.set_servos({3: VISOR_HALF, 1: HEADLIFT_UP, 2: HEADTILT_SLIGHT_DOWN})
    elif frame.affect == "happy":
        servos.set_servos({3: VISOR_OPEN, 1: HEADLIFT_UP})
    else:
        servos.set_servos({3: VISOR_HALF, 1: HEADLIFT_NEUTRAL})


def speech_stop() -> None:
    """Call when TTS finishes. Stops gesture thread and resets head pose to idle."""
    _speaking.clear()
    leds_head.speak_stop()
    leds_chest.idle()
    servos.set_breathing_emotion("neutral")
    baseline = servos.get_face_tracking_baseline()
    servos.set_servos({
        0: baseline.get(0, NECK_CENTER),
        3: VISOR_HALF,
        1: baseline.get(1, HEADLIFT_NEUTRAL),
        2: baseline.get(2, HEADTILT_NEUTRAL),
    })


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


def _face_tracking_holding_gaze() -> bool:
    try:
        for person in world_state.get("people") or []:
            if person.get("face_visible") is False or person.get("face_missing"):
                continue
            if person.get("face_box") or person.get("bounding_box") or person.get("bbox"):
                return True
    except Exception:
        pass

    tracking = _world_self_state().get("face_tracking") or {}
    if not isinstance(tracking, dict):
        return False
    if tracking.get("searching") or tracking.get("directed_hold"):
        return True
    return bool(
        tracking.get("locked")
        and (
            tracking.get("visible") is True
            or tracking.get("holding_lost_lock") is True
        )
    )


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


# A compound gaze is one pose on two axes: yaw on the neck channel, pitch on
# headlift+headtilt. Canonical form is "{pitch}_{yaw}" -- pitch first, always.
_COMPOUND_DIRECTED_LOOKS = {"down_left", "down_right", "up_left", "up_right"}


def _opposite_direction() -> str:
    # Mirror BOTH axes of a compound before consulting the neck read: the
    # opposite of "down and to the left" is up and to the right, but
    # _current_lateral_direction() only knows yaw and would answer a bare
    # "right", quietly leaving the camera pointed at the floor.
    last = (_last_directed_look or "").strip().lower()
    if last in _COMPOUND_DIRECTED_LOOKS:
        pitch, yaw = last.split("_")
        return "{}_{}".format(
            "up" if pitch == "down" else "down",
            "right" if yaw == "left" else "left",
        )
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
    elif norm.replace("-", "_").replace(" ", "_") in _COMPOUND_DIRECTED_LOOKS:
        # Accept a diagonal. Without this the whitelist below coerced
        # "down_left" to "current" and the head never moved at all -- the
        # second half of the 2026-08-13 dog-on-the-floor failure, and the
        # reason a compound could not simply be threaded through by the parser.
        norm = norm.replace("-", "_").replace(" ", "_")
    elif norm not in {"left", "right", "up", "down", "center", "current"}:
        norm = "current"

    settle = float(getattr(config, "DIRECTED_LOOK_SETTLE_SECS", 0.65))
    step_us = int(getattr(config, "DIRECTED_LOOK_STEP_QUS", 30))
    step_delay = float(getattr(config, "DIRECTED_LOOK_STEP_DELAY_SECS", 0.032))

    neck_cfg = config.SERVO_CHANNELS["neck"]
    lift_cfg = config.SERVO_CHANNELS["headlift"]
    tilt_cfg = config.SERVO_CHANNELS["headtilt"]
    visor_cfg = config.SERVO_CHANNELS["visor"]
    neck_ch = int(neck_cfg["ch"])
    lift_ch = int(lift_cfg["ch"])
    tilt_ch = int(tilt_cfg["ch"])
    visor_ch = int(visor_cfg["ch"])

    targets = {visor_ch: int(visor_cfg["max"])}
    # One pose, not two moves. A compound ("down_left") puts its yaw and its
    # pitch into the SAME move_to targets dict, so neck, headlift and headtilt
    # interpolate together in a single glide -- posing the axes as two
    # sequential calls would jerk the 5 lb head twice, which the inverted
    # headtilt on its 8mm rod must never see. Per-axis values are byte-identical
    # to the single-axis poses, so a diagonal never commands a tilt beyond what
    # a plain "down" already commands: down_left is exactly down's lift/tilt
    # plus left's neck.
    for part in (norm.split("_") if "_" in norm else [norm]):
        if part == "left":
            targets[neck_ch] = int(neck_cfg["min"])
        elif part == "right":
            targets[neck_ch] = int(neck_cfg["max"])
        elif part == "up":
            targets[lift_ch] = int(lift_cfg["max"])
            # Headtilt is inverted: lower values tilt the head/camera upward.
            targets[tilt_ch] = int(tilt_cfg["min"])
        elif part == "down":
            targets[lift_ch] = int(lift_cfg["min"])
            targets[tilt_ch] = int(tilt_cfg["max"])
        elif part == "center":
            targets.update({
                neck_ch: int(neck_cfg["neutral"]),
                lift_ch: int(lift_cfg["neutral"]),
                tilt_ch: int(tilt_cfg["neutral"]),
            })

    with _motion_lock:
        servos.move_to(targets, step_us=step_us, step_delay=step_delay)
        servos.set_face_tracking_baseline(
            neck=targets.get(neck_ch),
            lift=targets.get(lift_ch),
            tilt=targets.get(tilt_ch),
        )
        time.sleep(settle)
    _record_directed_look(norm, target)
    return norm


def travel_glance_pose(side: str = "center", pitch: str = "level",
                       fraction: float = 1.0) -> None:
    """One scenery glance while the base is rolling: neck yaw + head pitch in a
    single glide.

    directed_look_pose serves a user-directed inspection (full pitch range, long
    settle, recorded for "what do you see"); this one is for the exploration
    travel sweep. Pitch runs at HALF amplitude so the heavy head never rides the
    lift/tilt extremes on every swing of a drive, there is no settle sleep (the
    travel-gaze loop paces itself), and the glance is not recorded as a directed
    look.

    ``fraction`` scales how far toward the side the NECK travels (1.0 = full
    throw). Callers that need an interruptible sweep pose a side in fractional
    chunks with stop checks between them — a single full-throw glide blocks for
    seconds and cannot be aborted mid-flight (the come-search dwell sweep kept
    dragging the camera off a person it had just found, field 2026-08-11 19:57).
    """
    side = (side or "center").strip().lower()
    pitch = (pitch or "level").strip().lower()
    fraction = max(0.0, min(1.0, float(fraction)))

    neck_cfg = config.SERVO_CHANNELS["neck"]
    lift_cfg = config.SERVO_CHANNELS["headlift"]
    tilt_cfg = config.SERVO_CHANNELS["headtilt"]
    visor_cfg = config.SERVO_CHANNELS["visor"]
    neck_ch = int(neck_cfg["ch"])
    lift_ch = int(lift_cfg["ch"])
    tilt_ch = int(tilt_cfg["ch"])
    step_us = int(getattr(config, "DIRECTED_LOOK_STEP_QUS", 30))
    step_delay = float(getattr(config, "DIRECTED_LOOK_STEP_DELAY_SECS", 0.032))

    neck_neutral = int(neck_cfg["neutral"])
    targets = {int(visor_cfg["ch"]): int(visor_cfg["max"])}
    if side == "left":
        targets[neck_ch] = neck_neutral + int((int(neck_cfg["min"]) - neck_neutral) * fraction)
    elif side == "right":
        targets[neck_ch] = neck_neutral + int((int(neck_cfg["max"]) - neck_neutral) * fraction)
    else:
        targets[neck_ch] = neck_neutral

    lift_neutral = int(lift_cfg["neutral"])
    tilt_neutral = int(tilt_cfg["neutral"])
    if pitch == "up":
        # Headtilt is inverted: lower values tilt the head/camera upward.
        targets[lift_ch] = (lift_neutral + int(lift_cfg["max"])) // 2
        targets[tilt_ch] = (tilt_neutral + int(tilt_cfg["min"])) // 2
    elif pitch == "down":
        targets[lift_ch] = (lift_neutral + int(lift_cfg["min"])) // 2
        targets[tilt_ch] = (tilt_neutral + int(tilt_cfg["max"])) // 2
    elif pitch in ("down-slight", "slight-down"):
        # Approach gaze: the head stays at height, only the camera dips a touch
        # so floor clutter directly ahead is in frame while a standing person's
        # face still is (owner 2026-08-19: come-here should look slightly down).
        targets[lift_ch] = lift_neutral
        targets[tilt_ch] = HEADTILT_SLIGHT_DOWN
    else:
        targets[lift_ch] = lift_neutral
        targets[tilt_ch] = tilt_neutral

    with _motion_lock:
        servos.move_to(targets, step_us=step_us, step_delay=step_delay)
        servos.set_face_tracking_baseline(
            neck=targets.get(neck_ch),
            lift=targets.get(lift_ch),
            tilt=targets.get(tilt_ch),
        )


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


def wake_word_ack_wave(count: int | None = None, *, async_: bool = True) -> bool:
    """
    Brief wake-word recognition gesture.

    The hand rocks left/right while the elbow moves up/down in the same move_to
    calls, producing a compact "I heard that" wave without blocking listening.
    """
    if count is None:
        count = int(getattr(config, "WAKE_WORD_RECOGNITION_WAVE_COUNT", 3))
    count = max(1, min(6, int(count)))

    if async_:
        threading.Thread(
            target=_run_wake_word_ack_wave,
            args=(count,),
            daemon=True,
            name="wake_word_ack_wave",
        ).start()
        return True
    return _run_wake_word_ack_wave(count)


def _run_wake_word_ack_wave(count: int) -> bool:
    if not _body_beat_allowed():
        return False
    if not _arm_motion_lock.acquire(blocking=False):
        return False

    step_us = int(getattr(config, "WAKE_WORD_RECOGNITION_WAVE_STEP_QUS", 320))
    step_delay = float(getattr(config, "WAKE_WORD_RECOGNITION_WAVE_STEP_DELAY_SECS", 0.010))
    hold = float(getattr(config, "WAKE_WORD_RECOGNITION_WAVE_HOLD_SECS", 0.045))
    snapshot = _current_body_pose((4, 5, 7))

    servos.pause_arm_idle()
    try:
        with _motion_lock:
            servos.move_to(
                {7: HEROARM_FORWARD, 4: ELBOW_NEUTRAL, 5: HAND_NEUTRAL},
                step_us=step_us,
                step_delay=step_delay,
            )
            for _ in range(count):
                servos.move_to(
                    {4: ELBOW_UP, 5: HAND_RIGHT},
                    step_us=step_us,
                    step_delay=step_delay,
                )
                time.sleep(hold)
                servos.move_to(
                    {4: ELBOW_DOWN, 5: HAND_LEFT},
                    step_us=step_us,
                    step_delay=step_delay,
                )
                time.sleep(hold)
            servos.move_to(snapshot, step_us=step_us, step_delay=step_delay)
        return True
    except Exception as exc:
        _log.debug("[animations] wake-word ack wave failed: %s", exc)
        return False
    finally:
        servos.resume_arm_idle()
        _arm_motion_lock.release()


def wave_back_gesture(
    count: int | None = None, *, half_period: float | None = None, async_: bool = True
) -> bool:
    """Camera wave-back gesture.

    Eases the arm up to present the hand (a smooth, speed-limited raise from wherever
    the arm currently is — never a snap), then waves by sweeping the WRIST (the
    ``hand`` servo) between BOTH of its travel limits ``count`` times while the ELBOW
    bobs gently in sync — a clear hand wave, distinct from the compact wake-word ack.

    ``half_period`` (seconds per swing) overrides the configured default — used to mirror
    the user's wave speed (a smaller value = a faster wave-back).
    """
    if count is None:
        count = int(getattr(config, "WAVE_BACK_WRIST_SWEEPS", 4))
    count = max(1, min(8, int(count)))

    if async_:
        threading.Thread(
            target=_run_wave_back_gesture,
            args=(count,),
            kwargs={"half_period": half_period},
            daemon=True,
            name="wave_back_gesture",
        ).start()
        return True
    return _run_wave_back_gesture(count, half_period=half_period)


def _run_wave_back_gesture(count: int, half_period: float | None = None) -> bool:
    if not _body_beat_allowed():
        _log.info("[animations] wave-back skipped — state not active (sleep/shutdown)")
        return False
    # Wait briefly for the arm lock instead of instantly bailing: a transient idle-arm
    # wander move shouldn't silently swallow the wave-back. (Logged INFO so a "no wave"
    # report is diagnosable from the robot's INFO log.)
    if not _arm_motion_lock.acquire(timeout=1.5):
        _log.info("[animations] wave-back skipped — arm busy (couldn't get arm lock in 1.5s)")
        return False
    try:
        from audio import sound_effects
        # The wave-back IS a greeting — the R2-style greeting whistle fits it better than
        # a generic servo whir (cooldown owned by the effect layer).
        sound_effects.play("greeting")
    except Exception:
        pass

    # Sweep to the wrist servo's configured travel limits ("both direction maximums").
    # These are the safe limits from config/.env, so full travel is intentional and safe.
    hand_cfg = config.SERVO_CHANNELS.get("hand", {})
    hand_min = int(hand_cfg.get("min", HAND_LEFT))
    hand_max = int(hand_cfg.get("max", HAND_RIGHT))
    hand_range = max(1, abs(hand_max - hand_min))

    # The Maestro default speed (SERVO_DEFAULT_SPEED=40) is far too slow for the wrist's full
    # travel — a sweep takes ~2s at that rate, so rapid move_to reversals never complete (the
    # field symptom: one big swing then jitter, not 4 waves). Drive the wrist with DIRECT
    # targets at a fast speed instead, sleeping the travel time, then restore the defaults.
    if half_period is None:
        half_period = float(getattr(config, "WAVE_BACK_WRIST_HALF_PERIOD_SECS", 0.22))
    half_period = max(0.05, min(2.0, float(half_period)))
    configured_speed = int(getattr(config, "WAVE_BACK_WRIST_SPEED", 0))
    wave_accel = int(getattr(config, "WAVE_BACK_WRIST_ACCEL", 0))
    # Maestro speed unit ≈ 100 quarter-µs / second; auto-pick a speed that traverses the
    # wrist's full travel within half_period (full amplitude, ~no pause). 0 → auto.
    auto_speed = max(1, round(hand_range / (max(0.05, half_period) * 100.0)))
    wave_speed = configured_speed if configured_speed > 0 else auto_speed
    default_speed = int(getattr(config, "SERVO_DEFAULT_SPEED", 40))
    default_accel = int(getattr(config, "SERVO_DEFAULT_ACCELERATION", 8))

    step_us = int(getattr(config, "WAKE_WORD_RECOGNITION_WAVE_STEP_QUS", 320))
    step_delay = float(getattr(config, "WAKE_WORD_RECOGNITION_WAVE_STEP_DELAY_SECS", 0.010))
    snapshot = _current_body_pose((4, 5, 7))

    # Raise profile: a moderate, acceleration-limited glide from wherever the arm is up
    # to the presenting pose. The old code raised at the WAVE speed with accel 0
    # (unlimited) — the arm snapped to the pose from any starting position, which is the
    # hard jerk that opened every wave.
    raise_speed = max(1, int(getattr(config, "WAVE_BACK_RAISE_SPEED", 55)))
    raise_accel = int(getattr(config, "WAVE_BACK_RAISE_ACCEL", 14))
    # Elbow wave: bob between ELBOW_UP and ELBOW_UP + amplitude in sync with the wrist,
    # at its own speed sized to its (much smaller) travel so it eases rather than snaps.
    elbow_amp = max(0, int(getattr(config, "WAVE_BACK_ELBOW_WAVE_QUS", 340)))
    elbow_cfg = config.SERVO_CHANNELS.get("elbow", {})
    elbow_hi = min(int(elbow_cfg.get("max", ELBOW_DOWN)), ELBOW_UP + elbow_amp)
    elbow_speed = max(1, round(max(1, elbow_hi - ELBOW_UP) / (max(0.05, half_period) * 100.0)))
    elbow_accel = int(getattr(config, "WAVE_BACK_ELBOW_ACCEL", 20))

    _log.info(
        "[animations] wave-back gesture start: wrist(ch5) %d↔%d x%d, speed=%d accel=%d "
        "half_period=%.2fs elbow %d↔%d speed=%d raise speed=%d accel=%d "
        "servos_enabled=%s start_pose=%s",
        hand_min, hand_max, count, wave_speed, wave_accel, half_period,
        ELBOW_UP, elbow_hi, elbow_speed, raise_speed, raise_accel,
        getattr(servos, "SERVOS_ENABLED", "?"), snapshot,
    )
    arm_channels = (4, 5, 7)
    servos.pause_arm_idle()
    # Claim the arm: while this is set, the speech-reactive "talking with the hands" motion
    # leaves the arm channels alone (head/visor keep talking), so Rex's own greeting — or any
    # concurrent speech — can't override the wave. Without this the talking motion wins and
    # you see no wave.
    servos.begin_arm_gesture()
    try:
        with _motion_lock:
            try:
                # Phase 1 — smooth raise: glide the arm up at a moderate, accel-limited
                # profile, sleeping the actual travel time (Maestro speed unit ≈ 100
                # quarter-µs/s) so the wave starts only once the hand is presented.
                for ch in arm_channels:
                    servos.set_acceleration(ch, raise_accel)
                    servos.set_speed(ch, raise_speed)
                raise_targets = {7: HEROARM_FORWARD, 4: ELBOW_UP, 5: HAND_NEUTRAL}
                servos.set_servos(raise_targets)
                max_travel = max(
                    abs(int(raise_targets[ch]) - int(snapshot.get(ch, raise_targets[ch])))
                    for ch in raise_targets
                )
                raise_secs = max_travel / (raise_speed * 100.0) + 0.20
                time.sleep(min(1.5, max(0.25, raise_secs)))

                # Phase 2 — the wave: fast wrist sweeps limit-to-limit, elbow bobbing in
                # sync at its own gentler speed.
                servos.set_acceleration(5, wave_accel)
                servos.set_speed(5, wave_speed)
                servos.set_acceleration(4, elbow_accel)
                servos.set_speed(4, elbow_speed)
                for _ in range(count):
                    servos.set_servos({5: hand_max, 4: elbow_hi})
                    time.sleep(half_period)
                    servos.set_servos({5: hand_min, 4: ELBOW_UP})
                    time.sleep(half_period)
                servos.set_servos({5: HAND_NEUTRAL, 4: ELBOW_UP})
                time.sleep(half_period)
            finally:
                # Restore the channels' normal (slow, smooth) speed/accel.
                for ch in arm_channels:
                    servos.set_speed(ch, default_speed)
                    servos.set_acceleration(ch, default_accel)
            # Lower the arm back smoothly at the restored speed.
            servos.move_to(snapshot, step_us=step_us, step_delay=step_delay)
        _log.info("[animations] wave-back gesture done")
        return True
    except Exception as exc:
        _log.warning("[animations] wave-back gesture failed: %s", exc)
        return False
    finally:
        servos.end_arm_gesture()
        servos.resume_arm_idle()
        _arm_motion_lock.release()


# ---------------------------------------------------------------------------
# Composite reactions
# ---------------------------------------------------------------------------

def excited_burst() -> None:
    """Full excited reaction: arm up, head bob, visor open, chest AllRed."""
    leds_chest.speak("excited")
    leds_head.speak("excited")
    leds_head.set_eye_emotion("excited")
    servos.set_servos({3: VISOR_OPEN, 1: HEADLIFT_HIGH, 7: HEROARM_FORWARD, 4: ELBOW_UP})
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
