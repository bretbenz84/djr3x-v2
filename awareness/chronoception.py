"""
awareness/chronoception.py — Time and weather awareness for DJ-R3X.
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
from world_state import world_state

try:
    import requests as _requests
    _REQUESTS_OK = True
except ImportError:
    _REQUESTS_OK = False

_log = logging.getLogger(__name__)

# ─────────────────────────────────────────────────────────────────────────────
# Weather cache
# ─────────────────────────────────────────────────────────────────────────────

_weather_lock = threading.Lock()
_weather_cache: Optional[dict] = None
_weather_fetched_at: float = 0.0


def _weather_code_to_condition(code: int) -> str:
    if code == 113:
        return "clear"
    if code in (116, 119, 122):
        return "cloudy"
    if code in (143, 248, 260):
        return "fog"
    if code in (200, 386, 389, 392, 395):
        return "thunder"
    if code in (179, 227, 230, 320, 323, 326, 329, 332, 335, 338,
                350, 362, 365, 368, 371, 374, 377):
        return "snow"
    if 176 <= code <= 395:
        return "rain"
    return "unknown"


def _utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _weather_mood_bias(condition: str, temp_f: Optional[int]) -> tuple[str, str]:
    condition = (condition or "unknown").lower()
    if condition in {"thunder"}:
        return "stormy", "stormy weather adds theatrical, slightly keyed-up energy."
    if condition in {"rain"}:
        return "rainy", "rainy weather can make Rex a little drier and more atmospheric."
    if condition in {"snow"}:
        return "snowbound", "snowy weather makes cold-weather jokes and cozy contrast natural."
    if condition in {"fog"}:
        return "murky", "foggy weather supports mysterious, low-visibility banter."

    if temp_f is not None:
        if temp_f >= 95:
            return "heat-weary", "very hot weather can make Rex a touch more cranky and heat-dramatic."
        if temp_f >= 85:
            return "warm", "warm weather supports bright, energetic patter."
        if temp_f <= 40:
            return "cold-dramatic", "cold weather supports dry complaints about freezing circuits."
        if temp_f <= 55:
            return "cool", "cool weather supports crisp, lightly brisk observations."

    if condition == "clear":
        return "bright", "clear weather supports slightly brighter, upbeat energy."
    if condition == "cloudy":
        return "overcast", "cloudy weather supports a mildly dry, overcast mood."
    return "neutral", "weather is present but not strong enough to steer mood."


def _weather_unavailable(location: Optional[str] = None) -> dict:
    now = _utc_now_iso()
    return {
        "location": location or getattr(config, "WEATHER_LOCATION", None),
        "condition": "unknown",
        "temp_f": None,
        "feels_like_f": None,
        "humidity": None,
        "wind_mph": None,
        "description": "unavailable",
        "available": False,
        "source": "wttr.in",
        "fetched_at": now,
        "updated_at": now,
        "mood_bias": "unknown",
        "tone_hint": "weather feed is unavailable; do not invent conditions.",
    }


def _write_weather_to_world_state(weather: dict) -> None:
    try:
        current = world_state.get("weather")
    except Exception:
        current = {}
    current.update(dict(weather or {}))
    current["updated_at"] = current.get("updated_at") or _utc_now_iso()
    world_state.update("weather", current)


# ─────────────────────────────────────────────────────────────────────────────
# Background thread
# ─────────────────────────────────────────────────────────────────────────────

_stop_event = threading.Event()
_thread: Optional[threading.Thread] = None


# ─────────────────────────────────────────────────────────────────────────────
# Public API
# ─────────────────────────────────────────────────────────────────────────────

def get_time_context() -> dict:
    """
    Return current time context dict with keys:
    time_of_day, hour, day_of_week, is_weekend, notable_date, season.
    """
    now = datetime.now()
    hour = now.hour
    month = now.month
    day = now.day

    if 6 <= hour < 12:
        tod = "morning"
    elif 12 <= hour < 18:
        tod = "afternoon"
    elif 18 <= hour < 21:
        tod = "evening"
    elif 21 <= hour < 24:
        tod = "night"
    else:
        tod = "late_night"

    if month in (12, 1, 2):
        season = "winter"
    elif month in (3, 4, 5):
        season = "spring"
    elif month in (6, 7, 8):
        season = "summer"
    else:
        season = "autumn"

    return {
        "time_of_day": tod,
        "hour": hour,
        "day_of_week": now.strftime("%A"),
        "is_weekend": now.weekday() >= 5,
        "notable_date": config.NOTABLE_DATES.get((month, day)),
        "season": season,
    }


def fetch_weather(*, force: bool = False, update_world_state: bool = True) -> dict:
    """
    Fetch current weather from wttr.in for config.WEATHER_LOCATION.
    Cached for config.WEATHER_CACHE_SECS seconds.
    Returns dict with keys including condition, temp_f, description, and mood_bias.
    """
    global _weather_cache, _weather_fetched_at

    ttl = getattr(config, "WEATHER_CACHE_SECS", 600)

    with _weather_lock:
        if (
            not force
            and _weather_cache is not None
            and (time.monotonic() - _weather_fetched_at) < ttl
        ):
            cached = dict(_weather_cache)
            if update_world_state:
                _write_weather_to_world_state(cached)
            return cached

    if not _REQUESTS_OK:
        _log.warning("requests not available — weather fetch skipped")
        result = _weather_unavailable()
        with _weather_lock:
            _weather_cache = dict(result)
            _weather_fetched_at = time.monotonic()
        if update_world_state:
            _write_weather_to_world_state(result)
        return result

    location = config.WEATHER_LOCATION.replace(" ", "+")
    url = f"https://wttr.in/{location}?format=j1"
    try:
        resp = _requests.get(url, timeout=8)
        resp.raise_for_status()
        data = resp.json()
        cc = data["current_condition"][0]
        temp_f = int(cc["temp_F"])
        feels_like_f = int(cc["FeelsLikeF"]) if cc.get("FeelsLikeF") not in (None, "") else None
        humidity = int(cc["humidity"]) if cc.get("humidity") not in (None, "") else None
        wind_mph = int(cc["windspeedMiles"]) if cc.get("windspeedMiles") not in (None, "") else None
        description = cc["weatherDesc"][0]["value"]
        condition = _weather_code_to_condition(int(cc["weatherCode"]))
        mood_bias, tone_hint = _weather_mood_bias(condition, temp_f)
        now = _utc_now_iso()
        result = {
            "location": getattr(config, "WEATHER_LOCATION", None),
            "condition": condition,
            "temp_f": temp_f,
            "feels_like_f": feels_like_f,
            "humidity": humidity,
            "wind_mph": wind_mph,
            "description": description,
            "available": True,
            "source": "wttr.in",
            "fetched_at": now,
            "updated_at": now,
            "mood_bias": mood_bias,
            "tone_hint": tone_hint,
        }
        _log.info(
            "[chronoception] wttr.in returned for %s: %d°F %s (%s)",
            config.WEATHER_LOCATION, temp_f, description, condition,
        )
    except Exception as exc:
        _log.error("fetch_weather failed: %s", exc)
        result = _weather_unavailable(getattr(config, "WEATHER_LOCATION", None))

    with _weather_lock:
        _weather_cache = dict(result)
        _weather_fetched_at = time.monotonic()

    if update_world_state:
        _write_weather_to_world_state(result)

    return dict(result)


def refresh_weather(*, force: bool = False) -> dict:
    """Refresh weather and write the result into world_state.weather."""
    return fetch_weather(force=force, update_world_state=True)


def start_periodic_update(interval: Optional[float] = None) -> None:
    """Start a daemon thread that writes time and weather into WorldState."""
    global _thread, _stop_event

    if _thread and _thread.is_alive():
        return

    interval = interval or getattr(config, "CHRONOCEPTION_UPDATE_INTERVAL_SECS", 60.0)
    weather_interval = float(
        getattr(
            config,
            "WEATHER_UPDATE_INTERVAL_SECS",
            getattr(config, "WEATHER_CACHE_SECS", 600),
        )
    )
    _stop_event.clear()

    def _loop() -> None:
        next_weather_at = 0.0
        while not _stop_event.is_set():
            try:
                ctx = get_time_context()
                current = world_state.get("time")
                current.update(ctx)
                world_state.update("time", current)
                now = time.monotonic()
                if weather_interval >= 0 and now >= next_weather_at:
                    refresh_weather()
                    next_weather_at = now + max(1.0, weather_interval)
            except Exception as exc:
                _log.error("chronoception update failed: %s", exc)
            _stop_event.wait(interval)

    _thread = threading.Thread(target=_loop, daemon=True, name="chronoception")
    _thread.start()
    _log.info(
        "chronoception started (time_interval=%.1fs, weather_interval=%.1fs)",
        interval,
        weather_interval,
    )


def stop() -> None:
    """Stop the background update thread."""
    global _thread
    _stop_event.set()
    if _thread:
        _thread.join(timeout=5)
        _thread = None
    _log.info("chronoception stopped")
