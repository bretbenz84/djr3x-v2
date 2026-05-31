"""
emotion_orchestrator.py - shared emotional performance frames for DJ-R3X.

This module is deliberately side-effect-light and dependency-light. It turns
semantic emotion/event inputs into one compact frame that speech, motion, LEDs,
voice, and prompt wording can all obey.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
import time
from typing import Any, Optional

import config
from world_state import world_state


@dataclass(frozen=True)
class EmotionFrame:
    affect: str = "neutral"
    intensity: float = 0.35
    motion_style: str = "neutral"
    led_style: str = "neutral"
    voice_style: str = "default"
    word_style: str = "default_snark"
    body_beat: Optional[str] = None
    speech_motion: dict[str, float] = field(default_factory=dict)
    source: str = "default"
    trigger: Optional[str] = None

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)


_LED_STYLE_BY_AFFECT = {
    "neutral": "neutral",
    "curious": "curious",
    "happy": "happy",
    "excited": "excited",
    "starstruck": "excited",
    "giddy": "excited",
    "surprised": "excited",
    "startled": "excited",
    "sad": "sad",
    "sleepy": "sad",
    "angry": "angry",
    "annoyed": "angry",
    "disgusted": "angry",
    "sleep": "sleep",
}

_BASE_SPEECH_MOTION = {
    "interval_scale": 1.0,
    "head_wobble_mult": 1.0,
    "lift_wobble_mult": 1.0,
    "tilt_wobble_mult": 1.0,
    "arm_intensity_mult": 1.0,
    "hero_swing_mult": 1.0,
    "elbow_amp_mult": 1.0,
    "hand_amp_mult": 1.0,
    "lift_bias_qus": 0.0,
    "tilt_bias_qus": 0.0,
    "visor_open_floor_frac": 0.55,
    "visor_swing_mult": 1.0,
    "head_speed_mult": 1.0,
    "arm_speed_mult": 1.0,
    "breathing_period": "normal",
}

_PROFILES: dict[str, dict[str, Any]] = {
    "neutral": {
        "intensity": 0.35,
        "motion_style": "neutral",
        "voice_style": "default",
        "word_style": "default_snark",
        "speech_motion": {},
    },
    "curious": {
        "intensity": 0.50,
        "motion_style": "inquisitive",
        "voice_style": "default",
        "word_style": "curious_dry",
        "body_beat": "thinking_tilt",
        "speech_motion": {
            "interval_scale": 1.05,
            "head_wobble_mult": 0.85,
            "tilt_wobble_mult": 1.15,
            "arm_intensity_mult": 0.85,
            "tilt_bias_qus": -25.0,
            "visor_open_floor_frac": 0.58,
        },
    },
    "happy": {
        "intensity": 0.66,
        "motion_style": "buoyant",
        "voice_style": "warm",
        "word_style": "warm_snark",
        "body_beat": "happy_bounce",
        "speech_motion": {
            "interval_scale": 0.88,
            "head_wobble_mult": 1.12,
            "lift_wobble_mult": 1.18,
            "arm_intensity_mult": 1.08,
            "lift_bias_qus": 80.0,
            "visor_open_floor_frac": 0.68,
        },
    },
    "excited": {
        "intensity": 0.86,
        "motion_style": "animated",
        "voice_style": "energetic",
        "word_style": "high_energy",
        "body_beat": "giddy_wiggle",
        "speech_motion": {
            "interval_scale": 0.66,
            "head_wobble_mult": 1.35,
            "lift_wobble_mult": 1.35,
            "tilt_wobble_mult": 1.10,
            "arm_intensity_mult": 1.30,
            "hero_swing_mult": 1.20,
            "elbow_amp_mult": 1.20,
            "hand_amp_mult": 1.15,
            "lift_bias_qus": 140.0,
            "tilt_bias_qus": -45.0,
            "visor_open_floor_frac": 0.76,
            "visor_swing_mult": 1.18,
            "head_speed_mult": 1.25,
            "arm_speed_mult": 1.15,
            "breathing_period": "excited",
        },
    },
    "giddy": {
        "alias": "excited",
        "motion_style": "giddy",
        "word_style": "delighted",
    },
    "starstruck": {
        "alias": "excited",
        "intensity": 0.98,
        "motion_style": "celebrity_fan",
        "voice_style": "delighted",
        "word_style": "breathless_roast",
        "body_beat": "giddy_wiggle",
        "speech_motion": {
            "interval_scale": 0.52,
            "head_wobble_mult": 1.65,
            "lift_wobble_mult": 1.62,
            "tilt_wobble_mult": 1.22,
            "arm_intensity_mult": 1.65,
            "hero_swing_mult": 1.55,
            "elbow_amp_mult": 1.45,
            "hand_amp_mult": 1.35,
            "lift_bias_qus": 210.0,
            "tilt_bias_qus": -75.0,
            "visor_open_floor_frac": 0.88,
            "visor_swing_mult": 1.35,
            "head_speed_mult": 1.50,
            "arm_speed_mult": 1.35,
            "breathing_period": "excited",
        },
    },
    "surprised": {
        "intensity": 0.92,
        "motion_style": "startled",
        "voice_style": "startled",
        "word_style": "startled_short",
        "body_beat": "surprise_pop",
        "speech_motion": {
            "interval_scale": 0.60,
            "head_wobble_mult": 1.45,
            "lift_wobble_mult": 1.40,
            "tilt_wobble_mult": 1.05,
            "arm_intensity_mult": 0.95,
            "lift_bias_qus": 260.0,
            "tilt_bias_qus": -120.0,
            "visor_open_floor_frac": 0.92,
            "visor_swing_mult": 0.80,
            "head_speed_mult": 1.35,
            "arm_speed_mult": 1.05,
            "breathing_period": "excited",
        },
    },
    "sad": {
        "intensity": 0.48,
        "motion_style": "subdued",
        "voice_style": "calm",
        "word_style": "gentle",
        "body_beat": "sad_droop",
        "speech_motion": {
            "interval_scale": 1.55,
            "head_wobble_mult": 0.52,
            "lift_wobble_mult": 0.45,
            "tilt_wobble_mult": 0.55,
            "arm_intensity_mult": 0.35,
            "hero_swing_mult": 0.40,
            "elbow_amp_mult": 0.45,
            "hand_amp_mult": 0.40,
            "lift_bias_qus": -260.0,
            "tilt_bias_qus": 110.0,
            "visor_open_floor_frac": 0.50,
            "visor_swing_mult": 0.60,
            "head_speed_mult": 0.72,
            "arm_speed_mult": 0.65,
            "breathing_period": "sad",
        },
    },
    "angry": {
        "intensity": 0.78,
        "motion_style": "sharp",
        "voice_style": "clipped",
        "word_style": "clipped_irritated",
        "body_beat": "anger_flash",
        "speech_motion": {
            "interval_scale": 0.62,
            "head_wobble_mult": 0.95,
            "lift_wobble_mult": 0.85,
            "tilt_wobble_mult": 0.75,
            "arm_intensity_mult": 1.15,
            "hero_swing_mult": 0.95,
            "elbow_amp_mult": 1.25,
            "hand_amp_mult": 0.80,
            "lift_bias_qus": 90.0,
            "tilt_bias_qus": 140.0,
            "visor_open_floor_frac": 0.52,
            "visor_swing_mult": 0.55,
            "head_speed_mult": 1.45,
            "arm_speed_mult": 1.25,
        },
    },
    "disgusted": {
        "intensity": 0.72,
        "motion_style": "recoil",
        "voice_style": "repelled",
        "word_style": "disgusted_dry",
        "body_beat": "disgust_recoil",
        "speech_motion": {
            "interval_scale": 0.90,
            "head_wobble_mult": 0.70,
            "lift_wobble_mult": 0.65,
            "tilt_wobble_mult": 0.75,
            "arm_intensity_mult": 0.80,
            "lift_bias_qus": 80.0,
            "tilt_bias_qus": 170.0,
            "visor_open_floor_frac": 0.48,
            "visor_swing_mult": 0.55,
            "head_speed_mult": 1.10,
        },
    },
    "sleepy": {
        "alias": "sad",
        "intensity": 0.28,
        "motion_style": "drowsy",
        "word_style": "brief_low_energy",
    },
    "sleep": {
        "intensity": 0.10,
        "motion_style": "sleep",
        "voice_style": "quiet",
        "word_style": "minimal",
        "speech_motion": {
            "interval_scale": 1.8,
            "head_wobble_mult": 0.25,
            "lift_wobble_mult": 0.20,
            "tilt_wobble_mult": 0.25,
            "arm_intensity_mult": 0.10,
            "lift_bias_qus": -360.0,
            "tilt_bias_qus": 180.0,
            "visor_open_floor_frac": 0.35,
            "visor_swing_mult": 0.30,
            "head_speed_mult": 0.55,
            "arm_speed_mult": 0.45,
            "breathing_period": "sad",
        },
    },
}

_ALIASES = {
    "surprise": "surprised",
    "startled": "surprised",
    "shocked": "surprised",
    "joy": "giddy",
    "joyful": "giddy",
    "delighted": "giddy",
    "disgust": "disgusted",
    "grossed_out": "disgusted",
    "annoyed": "angry",
    "mad": "angry",
    "furious": "angry",
    "sleeping": "sleep",
}


def normalize_affect(value: str | None) -> str:
    key = "_".join(str(value or "neutral").strip().lower().replace("-", "_").split())
    if not key:
        return "neutral"
    return _ALIASES.get(key, key if key in _PROFILES else "neutral")


def frame_for_emotion(
    emotion: str | None,
    *,
    intensity: Optional[float] = None,
    source: str = "explicit",
    trigger: Optional[str] = None,
) -> EmotionFrame:
    affect = normalize_affect(emotion)
    profile = _resolve_profile(affect)
    motion = dict(_BASE_SPEECH_MOTION)
    motion.update(profile.get("speech_motion") or {})
    frame_intensity = _clamp01(
        profile.get("intensity", 0.35) if intensity is None else intensity
    )
    led_style = _LED_STYLE_BY_AFFECT.get(affect, _LED_STYLE_BY_AFFECT.get(profile.get("alias"), "neutral"))
    if led_style not in getattr(config, "EYE_COLORS", {}):
        led_style = "neutral"
    return EmotionFrame(
        affect=affect,
        intensity=frame_intensity,
        motion_style=str(profile.get("motion_style") or "neutral"),
        led_style=led_style,
        voice_style=str(profile.get("voice_style") or "default"),
        word_style=str(profile.get("word_style") or "default_snark"),
        body_beat=profile.get("body_beat"),
        speech_motion=motion,
        source=source,
        trigger=trigger,
    )


def frame_for_speech(emotion: str | EmotionFrame | dict | None) -> EmotionFrame:
    if isinstance(emotion, EmotionFrame):
        return emotion
    if isinstance(emotion, dict):
        return _frame_from_dict(emotion)
    return frame_for_emotion(emotion, source="speech")


def voice_settings_for_style(voice_style: str | None) -> Optional[dict[str, Any]]:
    """Map an emotion frame's voice_style to ElevenLabs voice_settings.

    Returns the configured baseline merged with any per-style deltas, so even
    neutral lines get an expressive baseline instead of the voice clone's flat
    defaults. Per-style dicts in config only need to list what differs from the
    baseline. Returns None when expressive voice is disabled, which makes TTS
    fall back to the clone's stored defaults (and the pre-existing cache).
    """
    if not getattr(config, "TTS_EXPRESSIVE_VOICE_ENABLED", True):
        return None
    baseline = dict(getattr(config, "TTS_VOICE_SETTINGS_BASELINE", {}) or {})
    style_key = str(voice_style or "default").strip().lower()
    overrides = (getattr(config, "TTS_VOICE_SETTINGS_BY_STYLE", {}) or {}).get(style_key)
    if isinstance(overrides, dict):
        baseline.update(overrides)
    return baseline or None


def voice_settings_for_emotion(
    emotion: str | EmotionFrame | dict | None,
) -> Optional[dict[str, Any]]:
    """Resolve ElevenLabs voice_settings for an emotion label/frame."""
    return voice_settings_for_style(frame_for_speech(emotion).voice_style)


def frame_for_empathy_mode(mode: str | None) -> EmotionFrame:
    mode_key = str(mode or "default").strip().lower()
    mapping = {
        "listen": "sad",
        "support": "sad",
        "acknowledge_then_yield": "sad",
        "ground": "sad",
        "course_correct": "sad",
        "crisis": "sad",
        "child_kind": "happy",
        "lift": "happy",
        "amplify": "excited",
        "validate": "curious",
        "gentle_probe": "curious",
        "brief": "sleepy",
        "kind_default": "neutral",
        "default": "neutral",
    }
    return frame_for_emotion(mapping.get(mode_key, "neutral"), source="empathy", trigger=mode_key)


def frame_for_event(event: str, **context: Any) -> EmotionFrame:
    key = "_".join(str(event or "").strip().lower().replace("-", "_").replace(".", "_").split())
    if key in {"surprise", "startle", "startled", "scream", "sudden_scream", "crash", "sudden_loud_sound"}:
        return frame_for_emotion("surprised", intensity=0.95, source="event", trigger=key)
    if key in {"animal_detected", "animal_arrival"}:
        species = str(context.get("species") or "").strip().lower()
        if is_startling_animal(species):
            return frame_for_emotion("surprised", intensity=0.95, source="event", trigger=f"animal:{species}")
        if "dog" in species:
            return frame_for_emotion("happy", intensity=0.70, source="event", trigger=f"animal:{species}")
        return frame_for_emotion("curious", intensity=0.62, source="event", trigger=f"animal:{species or 'unknown'}")
    if key in {"insult", "insult_detected"}:
        return frame_for_emotion("angry", intensity=0.80, source="event", trigger=key)
    if key in {"disgust", "gross"}:
        return frame_for_emotion("disgusted", intensity=0.78, source="event", trigger=key)
    return frame_for_emotion("neutral", source="event", trigger=key)


def is_startling_animal(species: str | None) -> bool:
    value = str(species or "").strip().lower()
    if not value:
        return False
    configured = getattr(config, "STARTLE_ANIMAL_SPECIES", None)
    if configured is None:
        configured = {
            "snake", "spider", "scorpion", "wasp", "hornet", "bee",
            "rat", "mouse", "bat", "lizard",
        }
    return any(token in value for token in configured)


def prompt_directive(frame: EmotionFrame) -> str:
    return (
        "Emotion performance frame: "
        f"affect={frame.affect}, intensity={frame.intensity:.2f}, "
        f"word_style={frame.word_style}, motion_style={frame.motion_style}, "
        f"voice_style={frame.voice_style}. Let word choice match this frame "
        "without announcing the frame."
    )


def publish_frame(frame: EmotionFrame, *, ttl_secs: float = 8.0) -> None:
    """Mirror the current emotional performance frame into world_state."""
    try:
        self_state = world_state.get("self_state")
        payload = frame.as_dict()
        payload["updated_at"] = time.time()
        payload["expires_at"] = time.time() + max(0.0, float(ttl_secs))
        self_state["emotion"] = frame.led_style if frame.led_style != "sleep" else "sleep"
        self_state["emotion_frame"] = payload
        self_state["body_state"] = frame.motion_style
        world_state.update("self_state", self_state)
    except Exception:
        pass


def current_frame(default: str = "neutral") -> EmotionFrame:
    try:
        frame = (world_state.get("self_state") or {}).get("emotion_frame") or {}
        if frame and float(frame.get("expires_at") or 0.0) >= time.time():
            return _frame_from_dict(frame)
    except Exception:
        pass
    return frame_for_emotion(default, source="current_default")


def _resolve_profile(affect: str) -> dict[str, Any]:
    profile = dict(_PROFILES.get(affect) or _PROFILES["neutral"])
    alias = profile.get("alias")
    if alias:
        base = _resolve_profile(str(alias))
        base.update({k: v for k, v in profile.items() if k != "alias"})
        return base
    return profile


def _frame_from_dict(data: dict[str, Any]) -> EmotionFrame:
    affect = normalize_affect(str(data.get("affect") or data.get("emotion") or "neutral"))
    base = frame_for_emotion(
        affect,
        intensity=data.get("intensity"),
        source=str(data.get("source") or "dict"),
        trigger=data.get("trigger"),
    )
    motion = dict(base.speech_motion)
    if isinstance(data.get("speech_motion"), dict):
        motion.update(data["speech_motion"])
    return EmotionFrame(
        affect=affect,
        intensity=_clamp01(data.get("intensity", base.intensity)),
        motion_style=str(data.get("motion_style") or base.motion_style),
        led_style=str(data.get("led_style") or base.led_style),
        voice_style=str(data.get("voice_style") or base.voice_style),
        word_style=str(data.get("word_style") or base.word_style),
        body_beat=data.get("body_beat", base.body_beat),
        speech_motion=motion,
        source=str(data.get("source") or base.source),
        trigger=data.get("trigger", base.trigger),
    )


def _clamp01(value: Any) -> float:
    try:
        return max(0.0, min(1.0, float(value)))
    except (TypeError, ValueError):
        return 0.0
