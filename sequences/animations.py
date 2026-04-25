"""
sequences/animations.py — Choreographed servo + LED sequences for DJ-R3X.

Each function coordinates hardware/servos.py, hardware/leds_head.py, and
hardware/leds_chest.py into a named, timed behavior.  All sequences are
synchronous; callers run them in threads as needed.

Background behaviors (headlift breathing, idle neck scan) run as daemon threads
via servos.breathing_thread() and servos.idle_animation() — those are not
duplicated here.  This module owns discrete triggered sequences only.

Emotion → LED pattern reference (encoded in chest Arduino firmware):
    neutral  → RandomBlocks2, normal brightness
    excited  → AllRed, full brightness (255)
    sad      → AllBlue, dim (55)
    angry    → rapid red strobe (255)
    happy    → confetti, normal brightness
"""

import threading
import time
import random

from hardware import servos, leds_head, leds_chest

# ---------------------------------------------------------------------------
# Servo position constants (Pololu quarter-microseconds)
# ---------------------------------------------------------------------------

# Ch 0 — Neck: 1984–9984, neutral 6000, right = higher
NECK_CENTER   = 6000
NECK_LEFT     = 4000
NECK_RIGHT    = 8000
NECK_FAR_LEFT = 2500
NECK_FAR_RIGHT = 9500

# Ch 1 — Headlift: 1984–7744, neutral 6000, higher = up
HEADLIFT_NEUTRAL = 6000
HEADLIFT_UP      = 4800
HEADLIFT_HIGH    = 3500
HEADLIFT_DOWN    = 7000
HEADLIFT_DROOP   = 7500

# Ch 2 — Headtilt: 3904–5504, neutral 4320, INVERTED (low value = head tilted up)
HEADTILT_NEUTRAL     = 4320
HEADTILT_UP          = 3904
HEADTILT_DOWN        = 5504
HEADTILT_SLIGHT_UP   = 4000
HEADTILT_SLIGHT_DOWN = 4700

# Ch 3 — Visor: 4544–6976, neutral 6000, higher = more open
VISOR_CLOSED  = 4544   # sleep / privacy — covers camera lens
VISOR_HALF    = 5760   # comfortable resting open
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
# Startup / shutdown
# ---------------------------------------------------------------------------

def startup() -> None:
    """Power-on: chest startup burst (ShortCircuit → RandomBlocks2), servos to neutral."""
    leds_chest.startup()
    # ShortCircuit animation runs for ~13 s on the Arduino then auto-switches to
    # CM_IDLE.  Fire an explicit IDLE command after that to guarantee the standard
    # blinking pattern is active regardless of firmware timing.
    def _chest_idle_after_startup():
        time.sleep(14)
        leds_chest.idle()
    threading.Thread(target=_chest_idle_after_startup, daemon=True, name="chest-startup-idle").start()
    leds_head.active()
    leds_head.set_eye_color(255, 200, 0)    # warm gold boot-up eyes
    servos.neutral(step_us=40)
    time.sleep(0.8)
    servos.set_servo(3, VISOR_HALF)         # visor opens to resting position
    time.sleep(0.4)


def shutdown() -> None:
    """Shutdown: head droops, visor closes, all LEDs off, then servos disconnect."""
    servos.set_servo(1, HEADLIFT_DOWN)
    servos.set_servo(2, HEADTILT_DOWN)
    servos.set_servo(3, VISOR_CLOSED)
    time.sleep(0.8)
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


# ---------------------------------------------------------------------------
# Speech
# ---------------------------------------------------------------------------

def speech_start(emotion: str = "neutral") -> None:
    """
    Call at the start of a TTS utterance.
    Sends the emotion pattern to both Arduinos and adjusts head pose to match.
    servos.set_breathing_emotion() is also updated so the breathing thread
    reflects the emotional state during speech.
    """
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
    """Call when TTS finishes. Resets mouth LEDs and head pose to idle."""
    leds_head.speak_stop()
    leds_chest.idle()
    servos.set_breathing_emotion("neutral")
    servos.set_servos({3: VISOR_HALF, 1: HEADLIFT_NEUTRAL, 2: HEADTILT_NEUTRAL})


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
