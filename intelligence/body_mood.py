"""
body_mood.py — Rex's sustained "body mood": a decaying affective posture that shapes
his physical body language (head lift/tilt bias, visor openness, idle micro-gestures)
between and around the primary face-tracking behavior.

This is PURE STATE + MAPPING. It imports no hardware and drives nothing on its own —
it is read by the consciousness motion layer (`_step_mood_expression`, the adaptive
rest-pose bias) which does the actual servo work, gated/failure-safe. Keeping it pure
makes it trivially testable and keeps the DAG clean (hardware depends on this idea, not
the other way around).

Design notes:
  • A mood is set by conversational events (complimented → proud, insulted → offended,
    amused → giddy) with an intensity (0..1) that DECAYS linearly to zero over a TTL, so
    Rex's posture relaxes back to neutral as the conversation moves on.
  • When no explicit event mood is active, an optional AMBIENT fallback derives a mild
    mood from the current emotion frame, so Rex still carries gentle posture.
  • The head bias rides on the REST pose (where the head settles when not tightly locked
    on a face) — it never fights the face-centering controller, honoring "tracking is
    primary, mood shapes posture."
  • The visor target is always kept at/above the lens-clear floor (VISOR_HALF) so a mood
    can never blind the camera Rex tracks faces with.
"""

from __future__ import annotations

import logging
import threading
import time
from typing import Optional

import config

_log = logging.getLogger(__name__)

# Visor lens-clear floor: VISOR_HALF (6400) is "default resting open — clear of camera
# lens" (sequences/animations.py). A mood visor target must never drop below it, or it
# would start covering the lens Rex uses for face tracking. Single-sourced from config
# so the visor RELEASE path (consciousness) and this target path share one safe floor.
_VISOR_LENS_CLEAR_FLOOR_DEFAULT = 6400


def visor_lens_clear_floor() -> int:
    try:
        return int(getattr(config, "BODY_MOOD_VISOR_LENS_CLEAR_FLOOR", _VISOR_LENS_CLEAR_FLOOR_DEFAULT))
    except Exception:
        return _VISOR_LENS_CLEAR_FLOOR_DEFAULT

# mood → (headlift_delta_qus, headtilt_delta_qus, visor_target_qus | None)
#   headlift: +up / -down (neutral 3600 since the 2026-08-19 gear rebuild, range 1984..7744)
#   headtilt: INVERTED — -chin-up / +chin-down (neutral 4320, range 3904..5504)
#   visor:    higher = more open (lens-clear floor 6400, max 6976); None = don't command
# Magnitudes are tuned "medium / playful": clearly readable, not constant motion.
_MOOD_POSE: dict[str, tuple[int, int, Optional[int]]] = {
    "proud":      (700, -180, 6976),   # head high, chin up, visor at MAX — praise
                                       # opens the visor all the way (was 6900)
    "giddy":      (650, -120, 6976),   # bouncy, visor wide open
    "amused":     (520, -110, 6850),
    "happy":      (450,  -90, 6700),
    "surprised":  (900, 0, 6976),      # head pops up, visor max — headtilt stays
                                       # parked (heavy head on the 8 mm tilt rod;
                                       # lift + visor carry the surprise)
    "curious":    (160, 200, 6500),    # slight lift, chin down, eyes a touch narrowed
    "thinking":   (120, 260, 6450),
    "suspicious": (90,  170, 6400),    # narrowed side-eye
    "annoyed":    (-250, 120, 6400),   # slight droop, subdued visor
    "offended":   (260, -260, 6400),   # haughty chin up, indignant — visor narrowed
                                       # to the floor (an offended squint, was open 6500)
    "angry":      (220, -200, 6400),   # alert, chin up, visor NARROWED to the floor —
                                       # a glare-squint, never an open 'glare' (was 6800).
                                       # The lens-clear floor caps it so it can't fully
                                       # cover the camera ("squint, but not blind").
    "sad":        (-700, 260, 6400),   # droop down, chin down, subdued (lens-clear)
    "bored":      (-450, 180, 6400),   # droop, subdued
    "neutral":    (0, 0, None),
}

# mood → an existing body-beat name (the 15 canonical beats) for an idle expression
_MOOD_BEAT: dict[str, Optional[str]] = {
    "proud":      "proud_dj_pose",
    "giddy":      "giddy_wiggle",
    "amused":     "giddy_wiggle",
    "happy":      "happy_bounce",
    "surprised":  "surprise_pop",
    "curious":    "thinking_tilt",
    "thinking":   "thinking_tilt",
    "suspicious": "suspicious_glance",
    "annoyed":    "offended_recoil",
    "offended":   "offended_recoil",
    "angry":      "anger_flash",
    "sad":        "sad_droop",
    "bored":      "thinking_tilt",
    "neutral":    None,
}

# Friendly aliases → canonical mood (mirrors performance_plan's mood-pose aliases).
_MOOD_ALIASES: dict[str, str] = {
    "complimented": "proud",
    "praised": "proud",
    "flattered": "proud",
    "smug": "proud",
    "insulted": "offended",
    "offense": "offended",
    "mad": "angry",
    "furious": "angry",
    "irritated": "annoyed",
    "fed_up": "annoyed",
    "delight": "giddy",
    "delighted": "giddy",
    "excited": "giddy",
    "joy": "giddy",
    "amusement": "amused",
    "funny": "amused",
    "skeptical": "suspicious",
    "shocked": "surprised",
    "startled": "surprised",
    "thoughtful": "thinking",
    "confused": "thinking",
    "down": "sad",
    "dejected": "sad",
}

# Ambient emotion-frame affect → a mild mood (used only when no event mood is active).
_AMBIENT_AFFECT_TO_MOOD: dict[str, str] = {
    "happy": "happy",
    "excited": "giddy",
    "starstruck": "giddy",
    "curious": "curious",
    "surprised": "surprised",
    "sad": "sad",
    "angry": "angry",
    "disgusted": "annoyed",
    "sleepy": "bored",
    "neutral": "neutral",
}

_state = {"mood": "neutral", "intensity": 0.0, "set_at": 0.0, "ttl": 0.0, "source": ""}
_lock = threading.Lock()


def _now() -> float:
    # Indirected so tests can patch the clock deterministically.
    return time.monotonic()


def enabled() -> bool:
    try:
        return bool(getattr(config, "BODY_LANGUAGE_MOOD_ENABLED", True))
    except Exception:
        return False


def canonical_mood(mood: Optional[str]) -> Optional[str]:
    key = "_".join(str(mood or "").strip().lower().replace("-", "_").split())
    if not key:
        return None
    key = _MOOD_ALIASES.get(key, key)
    return key if key in _MOOD_POSE else None


def set_mood(mood: str, *, intensity: float = 1.0, ttl: Optional[float] = None, source: str = "") -> bool:
    """Set Rex's current body mood. Returns True if accepted. A stronger/equal new mood
    replaces a weaker decaying one; a clearly weaker mood does not stomp a fresh strong
    one (so a passing 'curious' doesn't erase an active 'proud')."""
    if not enabled():
        return False
    canonical = canonical_mood(mood)
    if canonical is None or canonical == "neutral":
        return False
    intensity = max(0.0, min(1.0, float(intensity)))
    if intensity <= 0.0:
        return False
    if ttl is None:
        ttl = float(getattr(config, "BODY_MOOD_DEFAULT_TTL_SECS", 45.0))
    ttl = max(1.0, float(ttl))
    fresh_transition = False
    with _lock:
        current_int = _decayed_intensity_locked(_now())
        # Don't let a markedly weaker mood overwrite a still-strong active one.
        if _state["mood"] != "neutral" and _state["mood"] != canonical:
            if intensity < current_int - 0.25:
                return False
        fresh_transition = _state["mood"] != canonical
        _state.update(
            {"mood": canonical, "intensity": intensity, "set_at": _now(), "ttl": ttl, "source": str(source or "")}
        )
    if fresh_transition:
        _mood_chirp(canonical)
    return True


# A curated map of body-mood -> sound-effect key, for moods whose chirp is otherwise
# unreachable (the TTS emotion vocabulary never emits "proud"/"amused", so these
# expressive clips would sit unused). Fires ONCE on a fresh transition into the mood;
# the effect layer owns the cooldown so a flurry of compliments won't stack chirps.
# Overridable via config.SOUND_EFFECTS_MOOD_CHIRPS.
_MOOD_CHIRPS = {"proud": "proud", "amused": "laughing"}


def _mood_chirp(mood: str) -> None:
    try:
        overrides = getattr(config, "SOUND_EFFECTS_MOOD_CHIRPS", None)
        mapping = overrides if isinstance(overrides, dict) else _MOOD_CHIRPS
        key = mapping.get(mood)
        if not key:
            return
        # Day-mood congruence (field 2026-08-05 21:20: a triumphant "proud" chirp
        # played minutes after Rex told Bret he was feeling sluggish and down — the
        # expressive layer contradicted the self he'd just described out loud). On a
        # notably low-energy or negative day, the CELEBRATORY chirps stay in the
        # drawer; the body-mood posture itself still shifts (it's subtle), only the
        # loud legible fanfare is suppressed.
        if key in ("proud", "laughing") and bool(
            getattr(config, "REX_MOOD_GATES_CELEBRATORY_CHIRPS", True)
        ):
            try:
                from intelligence import rex_mood
                day = rex_mood.current()
                if day is not None and (
                    day.valence <= float(getattr(
                        config, "REX_MOOD_CHIRP_SUPPRESS_VALENCE", -0.2))
                    or rex_mood.effective_energy() <= float(getattr(
                        config, "REX_MOOD_CHIRP_SUPPRESS_ENERGY", 0.25))
                ):
                    _log.info(
                        "[body_mood] %s chirp suppressed — day mood %r is too "
                        "low for a fanfare", key, day.label,
                    )
                    return
            except Exception:
                pass
        from audio import sound_effects
        sound_effects.play(key)
    except Exception:
        pass


def _decayed_intensity_locked(now: float) -> float:
    ttl = float(_state.get("ttl") or 0.0)
    if ttl <= 0.0:
        return 0.0
    age = now - float(_state.get("set_at") or 0.0)
    if age <= 0.0:
        return float(_state.get("intensity") or 0.0)
    if age >= ttl:
        return 0.0
    return float(_state.get("intensity") or 0.0) * (1.0 - age / ttl)


def current_mood() -> tuple[str, float]:
    """Return (mood, intensity) — the active event mood (decayed), or a mild ambient mood
    derived from the current emotion frame, or ('neutral', 0.0)."""
    if not enabled():
        return ("neutral", 0.0)
    now = _now()
    with _lock:
        mood = str(_state.get("mood") or "neutral")
        decayed = _decayed_intensity_locked(now)
    if mood != "neutral" and decayed > 0.02:
        return (mood, decayed)
    return _ambient_mood()


def _ambient_mood() -> tuple[str, float]:
    if not bool(getattr(config, "BODY_MOOD_AMBIENT_FALLBACK_ENABLED", True)):
        return ("neutral", 0.0)
    try:
        from intelligence import emotion_orchestrator
        frame = emotion_orchestrator.current_frame("neutral")
        affect = getattr(frame, "affect", "neutral") or "neutral"
    except Exception:
        return ("neutral", 0.0)
    mood = _AMBIENT_AFFECT_TO_MOOD.get(str(affect).strip().lower(), "neutral")
    if mood == "neutral":
        return ("neutral", 0.0)
    intensity = max(0.0, min(1.0, float(getattr(config, "BODY_MOOD_AMBIENT_INTENSITY", 0.4))))
    return (mood, intensity) if intensity > 0.02 else ("neutral", 0.0)


def head_bias() -> tuple[int, int]:
    """(headlift_delta_qus, headtilt_delta_qus) for the current mood, scaled by intensity
    and a global config scale. (0, 0) for neutral. The consciousness rest-pose adds these
    to the head's settling target (clamped to servo limits there)."""
    mood, intensity = current_mood()
    lift_d, tilt_d, _ = _MOOD_POSE.get(mood, (0, 0, None))
    if intensity <= 0.0 or (lift_d == 0 and tilt_d == 0):
        return (0, 0)
    scale = float(getattr(config, "BODY_MOOD_HEAD_SCALE", 1.0)) * intensity
    return (int(round(lift_d * scale)), int(round(tilt_d * scale)))


def visor_target() -> Optional[int]:
    """Absolute visor target (quarter-µs) for the current mood, or None to leave the visor
    alone. Always kept at/above the lens-clear floor so a mood can't blind face-tracking."""
    if not bool(getattr(config, "BODY_MOOD_VISOR_ENABLED", True)):
        return None
    mood, intensity = current_mood()
    if intensity < float(getattr(config, "BODY_MOOD_VISOR_MIN_INTENSITY", 0.25)):
        return None
    _, _, target = _MOOD_POSE.get(mood, (0, 0, None))
    if target is None:
        return None
    # Interpolate from the lens-clear resting visor toward the mood target by intensity,
    # so a faint mood barely cracks the visor and a strong one fully expresses it.
    base = visor_lens_clear_floor()
    value = base + (int(target) - base) * max(0.0, min(1.0, intensity))
    return max(base, min(6976, int(round(value))))


def breathing_emotion() -> Optional[str]:
    """Map the current mood onto a breathing cadence ('excited'/'sad'/'neutral'), or None
    when there's nothing distinctive to express."""
    mood, intensity = current_mood()
    if intensity < 0.25 or mood == "neutral":
        return None
    if mood in ("proud", "giddy", "amused", "happy", "surprised", "angry"):
        return "excited"
    if mood in ("sad", "bored"):
        return "sad"
    return None


def idle_beat() -> Optional[str]:
    """An existing body-beat name expressing the current mood, for an occasional idle
    gesture — or None when neutral/too faint."""
    mood, intensity = current_mood()
    if intensity < float(getattr(config, "BODY_MOOD_IDLE_GESTURE_MIN_INTENSITY", 0.4)):
        return None
    return _MOOD_BEAT.get(mood)


def clear() -> None:
    with _lock:
        _state.update({"mood": "neutral", "intensity": 0.0, "set_at": 0.0, "ttl": 0.0, "source": ""})


def snapshot() -> dict:
    """Diagnostic snapshot (decayed)."""
    mood, intensity = current_mood()
    with _lock:
        raw = dict(_state)
    return {"mood": mood, "intensity": round(intensity, 3), "raw": raw}
