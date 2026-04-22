"""
intelligence/personality.py — TARS-style personality parameter management, emotion state,
mood decay, and session-based anger escalation for DJ-R3X.
"""

import logging
import sys
import threading
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

_PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

import config
from memory import database as db
from world_state import world_state

_log = logging.getLogger(__name__)

# ─────────────────────────────────────────────────────────────────────────────
# Module-level state — all in-memory, resets on process restart
# ─────────────────────────────────────────────────────────────────────────────

_lock = threading.Lock()

_anger_level: int = 0
_anger_last_incremented: Optional[float] = None  # time.monotonic() timestamp

# 1.0 when a non-neutral emotion is set; decays toward 0 → snaps to neutral
_mood_intensity: float = 0.0


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _sync_anger_level(level: int) -> None:
    """Mirror the current anger level into world_state for prompt assembly."""
    self_state = world_state.get("self_state")
    self_state["anger_level"] = level
    world_state.update("self_state", self_state)


# ─────────────────────────────────────────────────────────────────────────────
# Personality parameters — DB-backed, persistent across sessions
# ─────────────────────────────────────────────────────────────────────────────

def get_param(param_name: str) -> int:
    """Read current value from personality_settings DB; fall back to config.PERSONALITY_DEFAULTS."""
    row = db.fetchone(
        "SELECT value FROM personality_settings WHERE parameter = ?",
        (param_name,),
    )
    if row is not None:
        return row["value"]
    return config.PERSONALITY_DEFAULTS.get(param_name, 50)


def set_param(param_name: str, value: int, updated_by: str = "unknown") -> tuple[int, int]:
    """
    Clamp value to 0–100, upsert into personality_settings, return (old_value, new_value).
    """
    old_value = get_param(param_name)
    new_value = max(0, min(100, int(value)))
    db.execute(
        """
        INSERT INTO personality_settings (parameter, value, updated_at, updated_by)
        VALUES (?, ?, ?, ?)
        ON CONFLICT(parameter) DO UPDATE SET
            value      = excluded.value,
            updated_at = excluded.updated_at,
            updated_by = excluded.updated_by
        """,
        (param_name, new_value, _now_iso(), updated_by),
    )
    return (old_value, new_value)


def set_param_by_level(param_name: str, level_name: str, updated_by: str) -> tuple[int, int]:
    """
    Resolve a named level (e.g. 'high', 'off') through config.PERSONALITY_NAMED_LEVELS
    and write the resulting integer via set_param. Raises ValueError for unknown levels.
    """
    level_key = level_name.lower().strip()
    if level_key not in config.PERSONALITY_NAMED_LEVELS:
        raise ValueError(
            f"Unknown personality level {level_name!r}. "
            f"Valid levels: {sorted(config.PERSONALITY_NAMED_LEVELS)}"
        )
    return set_param(param_name, config.PERSONALITY_NAMED_LEVELS[level_key], updated_by)


def get_all_params() -> dict[str, int]:
    """Return a dict of all personality parameter current values, seeded with config defaults."""
    rows = db.fetchall("SELECT parameter, value FROM personality_settings")
    result = dict(config.PERSONALITY_DEFAULTS)
    for row in rows:
        result[row["parameter"]] = row["value"]
    return result


def generate_acknowledgment(param_name: str, old_value: int, new_value: int) -> str:
    """
    Ask Rex to acknowledge a parameter change in character. Must be called after set_param
    so the DB already reflects the new value — the assembled system prompt will carry the
    updated setting, making Rex's delivery naturally match the new level.
    """
    # Deferred import to avoid circular dependency at module load
    from intelligence.llm import get_response

    direction = "up" if new_value > old_value else "down"
    prompt = (
        f"Your {param_name} parameter has just been adjusted {direction} "
        f"from {old_value}/100 to {new_value}/100. "
        f"Acknowledge this in a single punchy in-character line. "
        f"Deliver the line itself at the new {param_name} level — "
        f"if {param_name} is now low, your delivery should reflect that. "
        f"No explanation. Just react."
    )
    try:
        return get_response(prompt)
    except Exception as exc:
        _log.error("generate_acknowledgment failed: %s", exc)
        return f"...{param_name} recalibrated to {new_value}. Systems updated."


# ─────────────────────────────────────────────────────────────────────────────
# Emotion state — world_state-backed
# ─────────────────────────────────────────────────────────────────────────────

_VALID_EMOTIONS: frozenset[str] = frozenset(config.EYE_COLORS.keys())


def get_emotion() -> str:
    """Return current emotion from world_state.self_state."""
    return world_state.get("self_state").get("emotion", "neutral")


def set_emotion(emotion: str) -> None:
    """
    Set current emotion in world_state.self_state. Valid emotions are the keys of
    config.EYE_COLORS. Resets mood intensity to 1.0 so decay has a full course to run.
    """
    global _mood_intensity
    if emotion not in _VALID_EMOTIONS:
        raise ValueError(
            f"Unknown emotion {emotion!r}. Valid emotions: {sorted(_VALID_EMOTIONS)}"
        )
    self_state = world_state.get("self_state")
    self_state["emotion"] = emotion
    world_state.update("self_state", self_state)
    with _lock:
        _mood_intensity = 0.0 if emotion == "neutral" else 1.0


# ─────────────────────────────────────────────────────────────────────────────
# Mood decay
# ─────────────────────────────────────────────────────────────────────────────

def apply_mood_decay(seconds_elapsed: float) -> None:
    """
    Reduce mood intensity toward zero at config.MOOD_DECAY_RATE_PER_MINUTE per minute.
    Snaps emotion to neutral in world_state when intensity bottoms out.
    Also clears anger escalation if config.ANGER_COOLDOWN_SECS has elapsed since the
    last increment.
    """
    global _mood_intensity, _anger_level, _anger_last_incremented

    if seconds_elapsed <= 0:
        return

    should_go_neutral = False
    anger_level_changed = False

    with _lock:
        if _mood_intensity > 0.0:
            decay = config.MOOD_DECAY_RATE_PER_MINUTE * (seconds_elapsed / 60.0)
            _mood_intensity = max(0.0, _mood_intensity - decay)
            if _mood_intensity == 0.0:
                should_go_neutral = True

        if (
            _anger_level > 0
            and _anger_last_incremented is not None
            and (time.monotonic() - _anger_last_incremented) >= config.ANGER_COOLDOWN_SECS
        ):
            _anger_level = 0
            _anger_last_incremented = None
            anger_level_changed = True

    if should_go_neutral:
        self_state = world_state.get("self_state")
        if self_state.get("emotion", "neutral") != "neutral":
            self_state["emotion"] = "neutral"
            world_state.update("self_state", self_state)

    if anger_level_changed:
        _sync_anger_level(0)


# ─────────────────────────────────────────────────────────────────────────────
# Anger escalation — in-memory only, resets on process restart
# ─────────────────────────────────────────────────────────────────────────────

def get_anger_level() -> int:
    """Return current session anger level (0–4), auto-resetting if cooldown has elapsed."""
    global _anger_level, _anger_last_incremented
    anger_level_changed = False
    with _lock:
        if (
            _anger_level > 0
            and _anger_last_incremented is not None
            and (time.monotonic() - _anger_last_incremented) >= config.ANGER_COOLDOWN_SECS
        ):
            _anger_level = 0
            _anger_last_incremented = None
            anger_level_changed = True
        current_level = _anger_level

    if anger_level_changed:
        _sync_anger_level(current_level)

    return current_level


def increment_anger(person_id: Optional[int] = None) -> int:
    """
    Increment anger level (capped at 4) and refresh the cooldown timer.
    If person_id is provided, increments that person's lifetime_insult_count in the DB.
    Returns the new anger level.
    """
    global _anger_level, _anger_last_incremented
    with _lock:
        _anger_level = min(4, _anger_level + 1)
        _anger_last_incremented = time.monotonic()
        new_level = _anger_level

    _sync_anger_level(new_level)

    if person_id is not None:
        db.execute(
            "UPDATE people SET lifetime_insult_count = lifetime_insult_count + 1 WHERE id = ?",
            (person_id,),
        )

    return new_level


def decrement_anger() -> int:
    """Decrement anger level (floor 0). Returns new anger level."""
    global _anger_level
    with _lock:
        _anger_level = max(0, _anger_level - 1)
        new_level = _anger_level

    _sync_anger_level(new_level)
    return new_level


def reset_anger() -> None:
    """Reset anger level to 0 and clear the cooldown timestamp."""
    global _anger_level, _anger_last_incremented
    with _lock:
        _anger_level = 0
        _anger_last_incremented = None

    _sync_anger_level(0)
