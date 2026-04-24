"""
awareness/holidays.py — Public holiday calendar (date.nager.at, no API key).

Fetches the full year of public holidays for config.HOLIDAY_COUNTRY_CODE on
demand and caches the result. Entries are categorized as 'major' (Christmas,
New Year, Easter, Thanksgiving) — meriting a ~30-day-out plans question — or
'minor' (other public holidays, the 3-day-weekend kind) — meriting a ~7-day-out
plans question. Both windows are configurable.

The dispatcher (intelligence/consciousness.py) calls upcoming_holidays() each
tick to discover holidays whose approach window currently includes today.
"""

import logging
import sys
import threading
from datetime import date, datetime, timedelta
from pathlib import Path
from typing import Optional

_PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

import config

_log = logging.getLogger(__name__)

# Holiday names (case-insensitive contains-match) that get the major window.
# Everything else from the API gets the minor window.
_MAJOR_HOLIDAY_KEYWORDS = (
    "christmas",
    "new year",
    "easter sunday",
    "thanksgiving",
)

# Per-year cache: {year: [holiday_dict, ...]}
_cache: dict[int, list[dict]] = {}
_cache_lock = threading.Lock()


def _classify(name: str) -> str:
    lowered = (name or "").lower()
    for kw in _MAJOR_HOLIDAY_KEYWORDS:
        if kw in lowered:
            return "major"
    return "minor"


def _fetch_year(year: int, country_code: str) -> list[dict]:
    """Fetch the full year from date.nager.at. Returns [] on any failure."""
    try:
        import requests
    except ImportError:
        _log.warning("[holidays] requests not available — skipping fetch")
        return []
    url = f"https://date.nager.at/api/v3/PublicHolidays/{year}/{country_code}"
    try:
        resp = requests.get(url, timeout=5)
        resp.raise_for_status()
        raw = resp.json()
    except Exception as exc:
        _log.error("[holidays] fetch failed for %s/%s: %s", year, country_code, exc)
        return []

    holidays = []
    for entry in raw or []:
        try:
            iso_date = entry["date"]
            name = entry.get("localName") or entry.get("name") or ""
            holidays.append({
                "date": iso_date,                  # "YYYY-MM-DD"
                "month_day": iso_date[5:],         # "MM-DD"
                "name": name,
                "tier": _classify(name),
            })
        except Exception:
            continue
    _log.info(
        "[holidays] fetched %d holidays for %s/%s",
        len(holidays), year, country_code,
    )
    return holidays


def get_holidays(year: int) -> list[dict]:
    """Return cached holiday list for the year, fetching on first miss."""
    country = getattr(config, "HOLIDAY_COUNTRY_CODE", "US")
    with _cache_lock:
        cached = _cache.get(year)
    if cached is not None:
        return cached
    fetched = _fetch_year(year, country)
    with _cache_lock:
        _cache[year] = fetched
    return fetched


def upcoming_holidays(today: Optional[date] = None) -> list[dict]:
    """
    Return holidays whose approach window currently includes today.

    Each result dict adds:
      'days_until' (int)
      'window' ('major' or 'minor')

    Window = days before the holiday during which Rex should ask plans.
    """
    today = today or date.today()
    major_window = getattr(config, "HOLIDAY_MAJOR_WINDOW_DAYS", 30)
    minor_window = getattr(config, "HOLIDAY_MINOR_WINDOW_DAYS", 7)

    pool = list(get_holidays(today.year))
    # Also include early-Jan holidays from next year if we're at year-end.
    if today.month == 12:
        pool.extend(get_holidays(today.year + 1))

    upcoming = []
    for h in pool:
        try:
            hd = datetime.strptime(h["date"], "%Y-%m-%d").date()
        except (KeyError, ValueError):
            continue
        delta = (hd - today).days
        window = major_window if h.get("tier") == "major" else minor_window
        if 0 <= delta <= window:
            upcoming.append({**h, "days_until": delta, "window": h.get("tier", "minor")})
    upcoming.sort(key=lambda r: r["days_until"])
    return upcoming


def days_until_birthday(birthday_md: str, today: Optional[date] = None) -> Optional[int]:
    """
    Given a birthday stored as 'MM-DD' (or any string starting with MM-DD),
    return days until the next occurrence (0 = today, 364 max). None on parse fail.
    """
    today = today or date.today()
    if not birthday_md or len(birthday_md) < 5:
        return None
    try:
        month = int(birthday_md[0:2])
        day = int(birthday_md[3:5])
    except ValueError:
        return None
    try:
        next_bd = date(today.year, month, day)
    except ValueError:
        # Feb 29 in a non-leap year — fall back to Mar 1
        if month == 2 and day == 29:
            next_bd = date(today.year, 3, 1)
        else:
            return None
    if next_bd < today:
        try:
            next_bd = date(today.year + 1, month, day)
        except ValueError:
            next_bd = date(today.year + 1, 3, 1)
    return (next_bd - today).days
