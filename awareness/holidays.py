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
import time
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
# A failed fetch for a non-US calendar must not become a permanent empty cache,
# but it also must not retry on every consciousness tick. The US fallback below
# keeps the normal robot deployment calendar-aware even while offline.
_fetch_retry_after: dict[int, float] = {}


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


def _nth_weekday(year: int, month: int, weekday: int, occurrence: int) -> date:
    """Return the Nth weekday in a month (Monday=0)."""
    first = date(year, month, 1)
    offset = (weekday - first.weekday()) % 7
    return first + timedelta(days=offset + 7 * (occurrence - 1))


def _last_weekday(year: int, month: int, weekday: int) -> date:
    """Return the last weekday in a month (Monday=0)."""
    if month == 12:
        cursor = date(year + 1, 1, 1) - timedelta(days=1)
    else:
        cursor = date(year, month + 1, 1) - timedelta(days=1)
    return cursor - timedelta(days=(cursor.weekday() - weekday) % 7)


def _us_fallback_holidays(year: int) -> list[dict]:
    """Small offline US federal-holiday calendar.

    The hosted calendar remains preferred. This covers the dates most useful for
    natural plans questions when the robot has no network; it is deliberately a
    practical fallback, not a replacement for a jurisdiction-specific service.
    """
    rows = (
        ("New Year's Day", date(year, 1, 1)),
        ("Martin Luther King Jr. Day", _nth_weekday(year, 1, 0, 3)),
        ("Washington's Birthday", _nth_weekday(year, 2, 0, 3)),
        ("Memorial Day", _last_weekday(year, 5, 0)),
        ("Juneteenth National Independence Day", date(year, 6, 19)),
        ("Independence Day", date(year, 7, 4)),
        ("Labor Day", _nth_weekday(year, 9, 0, 1)),
        ("Columbus Day", _nth_weekday(year, 10, 0, 2)),
        ("Veterans Day", date(year, 11, 11)),
        ("Thanksgiving Day", _nth_weekday(year, 11, 3, 4)),
        ("Christmas Day", date(year, 12, 25)),
    )
    return [
        {
            "date": day.isoformat(),
            "month_day": day.strftime("%m-%d"),
            "name": name,
            "tier": _classify(name),
        }
        for name, day in rows
    ]


def get_holidays(year: int) -> list[dict]:
    """Return cached holiday list for the year, fetching on first miss."""
    country = getattr(config, "HOLIDAY_COUNTRY_CODE", "US")
    with _cache_lock:
        cached = _cache.get(year)
        retry_after = _fetch_retry_after.get(year, 0.0)
    if cached is not None:
        return cached
    if time.monotonic() < retry_after:
        return []
    fetched = _fetch_year(year, country)
    if not fetched and str(country).strip().upper() == "US":
        fetched = _us_fallback_holidays(year)
        _log.warning(
            "[holidays] using local US fallback calendar for %s after fetch failure",
            year,
        )
    with _cache_lock:
        if fetched:
            _cache[year] = fetched
            _fetch_retry_after.pop(year, None)
        else:
            retry_secs = max(1.0, float(getattr(config, "HOLIDAY_FETCH_RETRY_SECS", 300.0)))
            _fetch_retry_after[year] = time.monotonic() + retry_secs
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


def _holiday_when_phrase(days_until: int, today: Optional[date] = None) -> str:
    """Human phrase for how far off a holiday is: 'today', 'tomorrow', 'this Friday'
    (within the coming week), else 'in N days'."""
    if days_until <= 0:
        return "today"
    if days_until == 1:
        return "tomorrow"
    today = today or date.today()
    if days_until <= 6:
        return f"this {(today + timedelta(days=days_until)).strftime('%A')}"
    return f"in {days_until} days"


def next_relevant_holiday(today: Optional[date] = None) -> Optional[dict]:
    """The soonest upcoming holiday Rex should be AWARE of, respecting the major/minor
    toggle (HOLIDAY_PLANS_INCLUDE_MINOR), or None. Adds a 'when' phrase ('this Friday').

    Single source of truth for "is a holiday coming up" — used both to surface
    awareness in the conversation prompt and to let an idle lull pivot to asking about
    holiday plans. Network-backed but cached; callers wrap in try/except.
    """
    include_minor = bool(getattr(config, "HOLIDAY_PLANS_INCLUDE_MINOR", False))
    for holiday in upcoming_holidays(today):
        if holiday.get("window") == "minor" and not include_minor:
            continue
        return {**holiday, "when": _holiday_when_phrase(int(holiday.get("days_until", 0)), today)}
    return None


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
