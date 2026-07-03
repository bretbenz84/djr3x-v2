"""Cross-session relationship trends — computed, never LLM-generated.

Real friends notice patterns: "that's three days in a row", "first time in a couple
weeks", "we always end up talking about volleyball". The per-session data already
exists (people.visit_count/first_seen/last_seen + one `conversations` row per
person-session with session_date and comma-separated topics); this module turns it
into (a) a compact prompt line for person context and (b) at most ONE human-shaped
cadence remark for greetings. Pure SQL + python — zero tokens.
"""

from __future__ import annotations

import logging
from datetime import datetime, timezone
from typing import Optional

import config
from memory import conversations as conv_db
from memory import people as people_db

_log = logging.getLogger(__name__)

# Cache per (person, day) so the trend text is computed once per session, not per turn.
_cache: dict[tuple[int, str], dict] = {}


def _parse_dt(value) -> Optional[datetime]:
    try:
        dt = datetime.fromisoformat(str(value))
        return dt.replace(tzinfo=timezone.utc) if dt.tzinfo is None else dt
    except (TypeError, ValueError):
        return None


def visit_stats(person_id: int) -> dict:
    """Aggregate visit cadence + recurring topics from existing rows. Cached per day."""
    now = datetime.now(timezone.utc)
    key = (int(person_id), now.date().isoformat())
    cached = _cache.get(key)
    if cached is not None:
        return cached

    stats: dict = {
        "total_visits": 0, "days_known": 0, "sessions_7d": 0, "sessions_30d": 0,
        "streak_days": 0, "gap_days": None, "recurring_topics": [],
    }
    try:
        person = people_db.get_person(int(person_id)) or {}
        stats["total_visits"] = int(person.get("visit_count") or 0)
        first_seen = _parse_dt(person.get("first_seen"))
        if first_seen:
            stats["days_known"] = max(0, (now - first_seen).days)
        last_seen = _parse_dt(person.get("last_seen"))
        if last_seen:
            stats["gap_days"] = (now - last_seen).total_seconds() / 86400.0

        history = conv_db.get_conversation_history(int(person_id), limit=60) or []
        session_days: set = set()
        topic_days: dict[str, set] = {}
        for row in history:
            dt = _parse_dt(row.get("session_date"))
            if not dt:
                continue
            age_days = (now - dt).total_seconds() / 86400.0
            day = dt.date()
            session_days.add(day)
            if age_days <= 7.0:
                stats["sessions_7d"] += 1
            if age_days <= 30.0:
                stats["sessions_30d"] += 1
            for topic in str(row.get("topics") or "").split(","):
                topic = topic.strip().lower()
                if topic:
                    topic_days.setdefault(topic, set()).add(day)

        # Consecutive-day streak ending today or yesterday.
        streak, day = 0, now.date()
        if day not in session_days:
            day = day.fromordinal(day.toordinal() - 1)
        while day in session_days:
            streak += 1
            day = day.fromordinal(day.toordinal() - 1)
        stats["streak_days"] = streak

        # Topics that came up on 3+ DIFFERENT days — the "we always end up on this" set.
        stats["recurring_topics"] = sorted(
            (t for t, days in topic_days.items() if len(days) >= 3),
            key=lambda t: -len(topic_days[t]),
        )[:3]
    except Exception as exc:
        _log.debug("[trends] visit_stats failed for %s: %s", person_id, exc)
    _cache[key] = stats
    return stats


def summarize_for_prompt(person_id: Optional[int]) -> str:
    """One compact line for the person context (~25 tokens). '' when there's no story."""
    if person_id is None:
        return ""
    s = visit_stats(int(person_id))
    bits: list[str] = []
    if s["total_visits"] >= 3:
        bits.append(f"visit #{s['total_visits'] + 1}")
    if s["streak_days"] >= 2:
        bits.append(f"{s['streak_days']} days in a row")
    elif s["sessions_7d"] >= 3:
        bits.append(f"{s['sessions_7d']} visits in the last week — they come around a lot")
    if s["recurring_topics"]:
        bits.append("topics you two keep landing on: " + ", ".join(s["recurring_topics"]))
    if not bits:
        return ""
    return (
        "Relationship trend (you genuinely notice this, like a friend would — mention it "
        "only when natural, never as a statistic): " + "; ".join(bits) + "."
    )


def cadence_hook(person_id: Optional[int]) -> Optional[tuple[str, str]]:
    """At most ONE human-shaped cadence observation for a greeting, or None.

    Returns (kind, phrase): e.g. ("streak", "third day in a row"),
    ("frequent", "4 visits this week"), ("medium_gap", "about 2 weeks").
    The mid-band gap fills the hole between recent-return (<48h) and
    long-absence (>=60d) where no greeting hook existed at all."""
    if person_id is None:
        return None
    s = visit_stats(int(person_id))
    if s["streak_days"] >= 2:
        n = s["streak_days"] + 1  # counting today
        word = {2: "second", 3: "third", 4: "fourth", 5: "fifth"}.get(n, f"{n}th")
        return ("streak", f"the {word} day in a row")
    if s["sessions_7d"] >= int(getattr(config, "TREND_FREQUENT_SESSIONS_7D", 4)):
        return ("frequent", f"{s['sessions_7d']} visits in the last week")
    gap = s["gap_days"]
    if gap is not None and 3.0 <= gap < float(getattr(config, "LONG_ABSENCE_THRESHOLD_DAYS", 60)):
        if gap >= 14:
            return ("medium_gap", f"about {round(gap / 7)} weeks")
        return ("medium_gap", f"about {int(round(gap))} days")
    return None
