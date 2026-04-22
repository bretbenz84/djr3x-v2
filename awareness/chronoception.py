"""
awareness/chronoception.py — Time and weather awareness for DJ-R3X.
"""

import logging
import sys
import threading
import time
from datetime import datetime
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


def fetch_weather() -> dict:
    """
    Fetch current weather from wttr.in for config.WEATHER_LOCATION.
    Cached for config.WEATHER_CACHE_SECS seconds.
    Returns dict with keys: condition, temp_f, description.
    """
    global _weather_cache, _weather_fetched_at

    ttl = getattr(config, "WEATHER_CACHE_SECS", 600)

    with _weather_lock:
        if _weather_cache is not None and (time.monotonic() - _weather_fetched_at) < ttl:
            return _weather_cache

    if not _REQUESTS_OK:
        _log.warning("requests not available — weather fetch skipped")
        return {"condition": "unknown", "temp_f": None, "description": "unavailable"}

    location = config.WEATHER_LOCATION.replace(" ", "+")
    url = f"https://wttr.in/{location}?format=j1"
    try:
        resp = _requests.get(url, timeout=8)
        resp.raise_for_status()
        data = resp.json()
        cc = data["current_condition"][0]
        temp_f = int(cc["temp_F"])
        description = cc["weatherDesc"][0]["value"]
        condition = _weather_code_to_condition(int(cc["weatherCode"]))
        result = {"condition": condition, "temp_f": temp_f, "description": description}
    except Exception as exc:
        _log.error("fetch_weather failed: %s", exc)
        result = {"condition": "unknown", "temp_f": None, "description": "unavailable"}

    with _weather_lock:
        _weather_cache = result
        _weather_fetched_at = time.monotonic()

    return result


def start_periodic_update(interval: Optional[float] = None) -> None:
    """Start a daemon thread that writes get_time_context() into world_state.time."""
    global _thread, _stop_event

    if _thread and _thread.is_alive():
        return

    interval = interval or getattr(config, "CHRONOCEPTION_UPDATE_INTERVAL_SECS", 60.0)
    _stop_event.clear()

    def _loop() -> None:
        while not _stop_event.is_set():
            try:
                ctx = get_time_context()
                current = world_state.get("time")
                current.update(ctx)
                world_state.update("time", current)
            except Exception as exc:
                _log.error("chronoception update failed: %s", exc)
            _stop_event.wait(interval)

    _thread = threading.Thread(target=_loop, daemon=True, name="chronoception")
    _thread.start()
    _log.info("chronoception started (interval=%.1fs)", interval)


def stop() -> None:
    """Stop the background update thread."""
    global _thread
    _stop_event.set()
    if _thread:
        _thread.join(timeout=5)
        _thread = None
    _log.info("chronoception stopped")
