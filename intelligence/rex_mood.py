"""
rex_mood.py — Rex's inner weather: ONE mood per local day, and it's his.

The gap this closes (owner gripe 2026-08-05): Rex had no self-state at all. Asked
"how are you?" — directly, or bounced back as "how about you?" after he asked first
— the model had nothing to answer FROM, and `REX_CORE_PROMPT` helpfully hands it the
phrase "systems nominal" as a droid tic. So every single time, forever: "operating
within normal parameters." A canned answer isn't a personality bug, it's a missing
input.

So: a mood is MINTED once per LOCAL day from what the day actually handed him, and
persisted. It

  * is seeded by real signals — the weather, whatever news he's been chewing on, a
    holiday, the hour he woke up, the state of his own hardware — falling back to
    plain chance, which is also how moods work,
  * DRIFTS as the day goes (a long quiet stretch flattens him, a good conversation
    lifts him, being insulted sharpens him), bounded so the day keeps its character,
  * survives a reboot: coming back up at 4pm resumes the mood he woke up with plus
    the day's drift, rather than re-rolling a brand-new droid, and
  * is anti-repeated across recent days, so he doesn't wake up "restless" three
    mornings running.

Deliberately NOT written into world_state self_state["emotion"]: the three existing
affect layers are all SHORT and reactive —

    emotion_orchestrator frame   ~8s   per-utterance performance frame
    body_mood                    ~45s  decaying posture
    personality._mood_intensity  ~10m  decays back to hard "neutral"

— and personality.apply_mood_decay() would stomp a day mood to neutral within ten
minutes. This is the BASELINE those ride on top of, held in its own state.

Surfacing (both live voices, or it would be invisible):
  1. lean_brain._system_prompt injects prompt_lines() — that is THE live path, and it
     covers replies AND directives (greetings, proactive lines) under ONE VOICE.
  2. llm.assemble_system_prompt injects prompt_section() for the classic fallback and
     the web-search prompt.
The rex_pov trap, avoided on purpose: rex_pov injects only into assemble_system_prompt,
so under LEAN_BRAIN_ENABLED its preoccupation never reaches a direct reply at all.

No LLM call and no network of its own — it reads signals other subsystems already
fetched, so it runs under the test suite unmodified. Gated by config.REX_MOOD_ENABLED.

Related: intelligence/rex_pov.py (what he's chewing on) is the CONTENT counterpart to
this AFFECT layer; intelligence/greeting_cadence.py consumes the same "don't repeat
yourself at a returning human" instinct on the greeting side.
"""

from __future__ import annotations

import json
import logging
import random
import re
import threading
import time
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Optional

_log = logging.getLogger(__name__)


# ─────────────────────────────────────────────────────────────────────────────
# State
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class DayMood:
    date: str                       # local YYYY-MM-DD this mood belongs to
    seed_id: str
    label: str
    base_valence: float
    base_energy: float
    line: str                       # authored example answer, in Rex's voice
    because: str                    # what the day handed him, in Rex's voice
    seed_kind: str                  # weather | news | occasion | clock | self | chance
    minted_at: str                  # local ISO, seconds
    drift_valence: float = 0.0
    drift_energy: float = 0.0
    events: list = field(default_factory=list)   # drift kinds applied, for telemetry
    spoken: int = 0                 # times he has actually told someone how he is

    @property
    def valence(self) -> float:
        return _clamp(self.base_valence + self.drift_valence, -1.0, 1.0)

    @property
    def energy(self) -> float:
        return _clamp(self.base_energy + self.drift_energy, 0.0, 1.0)


_lock = threading.RLock()
_current: Optional[DayMood] = None
_recent_ids: list = []              # [{"date": str, "seed_id": str}, ...] newest last
_last_note_at: dict = {}            # drift kind -> monotonic stamp (per-kind pacing)


def _clamp(value: float, low: float, high: float) -> float:
    return max(low, min(high, float(value)))


# ─────────────────────────────────────────────────────────────────────────────
# Config accessors — read lazily so edits / test monkeypatching take effect
# ─────────────────────────────────────────────────────────────────────────────

def _enabled() -> bool:
    try:
        import config
        return bool(getattr(config, "REX_MOOD_ENABLED", True))
    except Exception:
        return False


def _seeds() -> list:
    """Normalized seed pool. Skips malformed rows rather than raising on a typo."""
    try:
        import config
        raw = getattr(config, "REX_MOOD_SEEDS", None) or []
    except Exception:
        return []
    out = []
    for seed in raw:
        if not isinstance(seed, dict):
            continue
        sid = str(seed.get("id") or "").strip()
        line = str(seed.get("line") or "").strip()
        if not sid or not line:
            continue
        fits = seed.get("fits") or ["any"]
        if isinstance(fits, str):
            fits = [fits]
        out.append({
            "id": sid,
            "label": str(seed.get("label") or sid).strip(),
            "valence": _clamp(seed.get("valence", 0.0), -1.0, 1.0),
            "energy": _clamp(seed.get("energy", 0.5), 0.0, 1.0),
            "line": line,
            "fits": tuple(str(f).strip().lower() for f in fits if str(f).strip()) or ("any",),
        })
    return out


def _drift_table() -> dict:
    try:
        import config
        table = getattr(config, "REX_MOOD_DRIFT", None) or {}
        return table if isinstance(table, dict) else {}
    except Exception:
        return {}


def _drift_limit() -> float:
    try:
        import config
        return abs(float(getattr(config, "REX_MOOD_DRIFT_LIMIT", 0.35) or 0.0))
    except Exception:
        return 0.35


def _drift_min_interval() -> float:
    try:
        import config
        return max(0.0, float(
            getattr(config, "REX_MOOD_DRIFT_MIN_INTERVAL_SECS", 600.0) or 0.0))
    except Exception:
        return 600.0


def _recent_memory_days() -> int:
    try:
        import config
        return max(0, int(getattr(config, "REX_MOOD_RECENT_MEMORY_DAYS", 5) or 0))
    except Exception:
        return 5


def _late_hour_drop() -> float:
    try:
        import config
        return abs(float(getattr(config, "REX_MOOD_LATE_HOUR_ENERGY_DROP", 0.2) or 0.0))
    except Exception:
        return 0.2


def _today() -> str:
    return datetime.now().strftime("%Y-%m-%d")


# ─────────────────────────────────────────────────────────────────────────────
# Day signals → tags
# ─────────────────────────────────────────────────────────────────────────────
#
# Every reader below is best-effort and returns nothing rather than raising: at boot
# most of them ARE empty (weather lands 1-3s after the chronoception thread starts;
# a cold-start news cache is empty until a ~45s web_search call returns). A mood
# minted with no signals at all is still a real mood — that's the `chance` path, and
# enrich() upgrades its *explanation* later if a richer signal shows up before he has
# actually told anyone how he is.

_WEATHER_TAGS = {
    "stormy":        (("charged", "edgy"), "there's a storm rolling through outside"),
    "rainy":         (("low", "inward"), "it's been raining all day"),
    "snowbound":     (("low", "cozy"), "it's snowing out there"),
    "murky":         (("inward", "flat"), "it's fogged in outside"),
    "heat-weary":    (("edgy", "drained"), "it is absurdly hot out"),
    "warm":          (("bright", "loose"), "it's warm out"),
    "cold-dramatic": (("edgy", "low"), "it's freezing out and my servos have opinions"),
    "cool":          (("crisp",), "it's crisp out"),
    "bright":        (("bright", "up"), "it's clear out"),
    "overcast":      (("flat",), "it's grey out"),
}


def _weather_signal(now: Optional[datetime] = None, allow_blocking: bool = False) -> tuple:
    """(tags, because) from the weather feed, or ((), '') when it hasn't landed.

    Reads the cached world_state copy rather than calling chronoception.fetch_weather()
    — that one BLOCKS on an 8s wttr.in GET, and minting a mood is not worth stalling
    boot for. Absent weather simply doesn't vote.
    """
    try:
        from world_state import world_state
        weather = world_state.snapshot().get("weather") or {}
        if not weather.get("available"):
            return ((), "")
        bias = str(weather.get("mood_bias") or "").strip().lower()
        tags, because = _WEATHER_TAGS.get(bias, ((), ""))
        return (tuple(tags), because)
    except Exception as exc:
        _log.debug("[rex_mood] weather signal skipped: %s", exc)
        return ((), "")


def _news_signal(now: Optional[datetime] = None, allow_blocking: bool = False) -> tuple:
    """(tags, because) from the day's cached stories. Pure disk read — no network."""
    try:
        from awareness import current_events
        stories = current_events.stories() or []
        if not stories:
            return ((), "")
        headline = str((stories[0] or {}).get("headline") or "").strip()
        if not headline:
            return ((), "")
        if len(headline) > 110:
            headline = headline[:107].rstrip() + "..."
        return (("chewing", "inward"), f"you've had \"{headline}\" rattling around all day")
    except Exception as exc:
        _log.debug("[rex_mood] news signal skipped: %s", exc)
        return ((), "")


def _occasion_signal(now: Optional[datetime] = None, allow_blocking: bool = False) -> tuple:
    """(tags, because) for a holiday landing today or tomorrow.

    The ONLY reader here that can block: holidays.get_holidays() does a 5s-timeout
    network fetch on the first miss per year, and the mint runs on the FOREGROUND boot
    path (before consciousness.start()). So it abstains unless the year cache is
    already warm — or unless the caller is the background enrich thread, which can
    afford to pay for the fetch.
    """
    try:
        from awareness import holidays
        if not allow_blocking:
            with holidays._cache_lock:
                warm = holidays._cache.get((now or datetime.now()).year) is not None
            if not warm:
                return ((), "")
        nxt = holidays.next_relevant_holiday()
        if not nxt:
            return ((), "")
        days = int(nxt.get("days_until") or 0)
        name = str(nxt.get("name") or "").strip()
        if not name or days > 1:
            return ((), "")
        when = "today" if days == 0 else "tomorrow"
        return (("occasion", "up"), f"it's {name} {when}")
    except Exception as exc:
        _log.debug("[rex_mood] occasion signal skipped: %s", exc)
        return ((), "")


def _clock_signal(now: Optional[datetime] = None, allow_blocking: bool = False) -> tuple:
    """(tags, because) from the calendar — the free signal that is always present."""
    now = now or datetime.now()
    weekday = now.weekday()
    hour = now.hour
    if hour < 6:
        return (("drained", "inward"), "you came up in the small hours and it shows")
    if weekday == 0:
        return (("flat",), "it's a Monday, which you maintain is a design flaw")
    if weekday == 4 and hour >= 15:
        return (("up", "loose"), "it's Friday afternoon and even you can feel it")
    if weekday >= 5:
        return (("loose", "cozy"), "it's the weekend and nobody's in a hurry")
    return ((), "")


def _self_signal(now: Optional[datetime] = None, allow_blocking: bool = False) -> tuple:
    """(tags, because) from his own hardware. Battery only speaks when it has a real
    reading — with no drive base attached current_mv() is -1 and tier_for_mv() returns
    None, which must read as "no opinion", never as a flat battery."""
    try:
        from intelligence import battery_awareness
        tier = battery_awareness.tier_for_mv(battery_awareness.current_mv())
        if tier == "low":
            return (("drained", "low"), "your pack is running down and you can feel it")
        if tier == "critical":
            return (("drained", "edgy"), "you are genuinely low on power")
        if tier == "charging":
            return (("bright",), "you're plugged in and topping up, which is its own kind of nice")
    except Exception as exc:
        _log.debug("[rex_mood] self signal skipped: %s", exc)
    return ((), "")


# Order matters: the FIRST signal that fires owns `because` (the most interesting
# explanation wins), but every signal that fires contributes its tags to the roll.
_SIGNALS = (
    ("occasion", _occasion_signal),
    ("news", _news_signal),
    ("weather", _weather_signal),
    ("self", _self_signal),
    ("clock", _clock_signal),
)


def _day_tags(now: Optional[datetime] = None, allow_blocking: bool = False) -> tuple:
    """(tags, because, seed_kind) for right now. `now` is injectable for tests.

    Every reader takes (now, allow_blocking) so the dispatch stays uniform, and a
    reader that raises is simply skipped — a broken feed must never cost Rex his mood.
    `allow_blocking` is False on the foreground boot path and True only from the
    background enrich thread.
    """
    tags: list = []
    because = ""
    kind = "chance"
    for name, reader in _SIGNALS:
        try:
            got_tags, got_because = reader(now, allow_blocking)
        except Exception:
            continue
        if got_tags:
            tags.extend(got_tags)
        if got_because and not because:
            because = got_because
            kind = name
    return (tuple(dict.fromkeys(tags)), because, kind)


# ─────────────────────────────────────────────────────────────────────────────
# Minting
# ─────────────────────────────────────────────────────────────────────────────

def _rng(date: str) -> random.Random:
    """Day-seeded RNG: the same day always rolls the same way, so a crash-and-restart
    before the state file is written resumes the SAME mood instead of re-rolling —
    and tests are reproducible without patching random."""
    return random.Random(f"rex-mood|{date}")


def _choose(tags: tuple, date: str, recent_ids: list) -> Optional[dict]:
    """Pick today's mood: recently-used moods are EXCLUDED, then a weighted roll over
    what's left (base 1, +2 per matching day-tag) picks from the survivors.

    Anti-repeat is an exclusion, not a de-weighting, for a concrete reason: the RNG is
    seeded on the DATE, so a given day always draws the same uniform value. Merely
    shrinking a weight slides the cumulative bands under a FIXED cursor instead of
    resampling — which reliably lands back on the de-weighted seed (measured: three
    identical moods in a row from a fresh state). Excluding the candidate outright is
    also what rex_pov._choose does, and it makes the guarantee exact.

    If exclusion empties the field (the memory window is as large as the pool), it
    recycles to the full pool rather than returning nothing.
    """
    seeds = _seeds()
    if not seeds:
        return None
    recent = {str(row.get("seed_id")) for row in recent_ids if isinstance(row, dict)}
    candidates = [s for s in seeds if s["id"] not in recent] or list(seeds)
    weights = [
        1.0 + 2.0 * sum(1 for tag in s["fits"] if tag != "any" and tag in tags)
        for s in candidates
    ]
    try:
        return _rng(date).choices(candidates, weights=weights, k=1)[0]
    except Exception:
        return candidates[0]


def _mint(date: str, now: Optional[datetime] = None) -> Optional[DayMood]:
    """Build today's mood. Caller must hold _lock."""
    tags, because, kind = _day_tags(now)
    seed = _choose(tags, date, _recent_ids)
    if seed is None:
        return None
    mood = DayMood(
        date=date,
        seed_id=seed["id"],
        label=seed["label"],
        base_valence=seed["valence"],
        base_energy=seed["energy"],
        line=seed["line"],
        because=because,
        seed_kind=kind,
        minted_at=(now or datetime.now()).isoformat(timespec="seconds"),
    )
    _remember(date, seed["id"])
    _log.info(
        "[rex_mood] minted %r for %s (valence=%+.2f energy=%.2f via=%s tags=%s)",
        mood.label, date, mood.base_valence, mood.base_energy, kind, sorted(tags),
    )
    return mood


def _remember(date: str, seed_id: str) -> None:
    """Append to the recent-mood memory, trimming to the configured window."""
    global _recent_ids
    _recent_ids = [row for row in _recent_ids
                   if isinstance(row, dict) and row.get("date") != date]
    _recent_ids.append({"date": date, "seed_id": seed_id})
    keep = _recent_memory_days()
    if keep and len(_recent_ids) > keep:
        _recent_ids = _recent_ids[-keep:]


# ─────────────────────────────────────────────────────────────────────────────
# Public API — read
# ─────────────────────────────────────────────────────────────────────────────

def ensure_today(now: Optional[datetime] = None) -> Optional[DayMood]:
    """The canonical accessor: mint today's mood if it isn't minted yet, else return
    the one already in hand (drift included). Cheap and idempotent — safe to call from
    any thread on any turn."""
    if not _enabled():
        return None
    global _current
    date = (now or datetime.now()).strftime("%Y-%m-%d")
    with _lock:
        if _current is not None and _current.date == date:
            return _current
        _current = _mint(date, now)
        return _current


def current() -> Optional[DayMood]:
    """Today's mood WITHOUT minting one. Pure read — for telemetry and tests."""
    with _lock:
        return _current


def effective_energy(now: Optional[datetime] = None) -> float:
    """Energy with the late-hour taper applied. Read-time only and never stored — he
    shouldn't still claim to be wired at 1am because he woke up wired at 9."""
    mood = ensure_today(now)
    if mood is None:
        return 0.5
    hour = (now or datetime.now()).hour
    if hour >= 22 or hour < 6:
        return _clamp(mood.energy - _late_hour_drop(), 0.0, 1.0)
    if hour >= 20:
        return _clamp(mood.energy - _late_hour_drop() / 2.0, 0.0, 1.0)
    return mood.energy


def _shade(mood: DayMood, energy: float) -> str:
    """How the day has gone SINCE he woke up — the part that makes an afternoon answer
    differ from a morning one even though the mood is the same."""
    dv, de = mood.drift_valence, mood.drift_energy
    if abs(dv) < 0.06 and abs(de) < 0.06:
        return ""
    if dv >= 0.06:
        return "and the day has been improving it"
    if dv <= -0.06:
        return "and the day has been chipping away at it"
    if de <= -0.06 or energy < 0.25:
        return "and you're running lower than you started"
    return "and you've picked up steam since"


def describe(now: Optional[datetime] = None) -> Optional[dict]:
    """Structured view of the mood — for the GUI, telemetry, and tests."""
    mood = ensure_today(now)
    if mood is None:
        return None
    with _lock:
        return {
            "date": mood.date,
            "seed_id": mood.seed_id,
            "label": mood.label,
            "valence": round(mood.valence, 3),
            "energy": round(effective_energy(now), 3),
            "base_valence": mood.base_valence,
            "base_energy": mood.base_energy,
            "because": mood.because,
            "seed_kind": mood.seed_kind,
            "line": mood.line,
            "minted_at": mood.minted_at,
            "events": list(mood.events),
            "spoken": mood.spoken,
        }


def prompt_lines(now: Optional[datetime] = None) -> list:
    """The lean-brain bullets. Kept to ONE line (~60 tokens) because this rides on
    EVERY lean call — reply and directive alike. Returns [] when disabled/unminted."""
    mood = ensure_today(now)
    if mood is None:
        return []
    energy = effective_energy(now)
    because = f" — {mood.because}" if mood.because else ""
    shade = _shade(mood, energy)
    shade_clause = f", {shade}" if shade else ""
    return [
        f"YOUR OWN STATE TODAY: you're {mood.label}{because}{shade_clause}. In your "
        f"own words it's roughly \"{mood.line}\" — that is an EXAMPLE of the shape, "
        f"never a script: if they ask how you are (including bouncing your own "
        f"question back at you), answer from this state freshly, in a new way each "
        f"time. Never \"systems nominal\" / \"normal parameters\" / an uptime figure. "
        f"Don't announce your mood unprompted or let it dominate — it colors how you "
        f"land, it isn't the topic."
    ]


def is_notable(now: Optional[datetime] = None) -> bool:
    """True when today's mood is far enough from the middle to be worth MENTIONING.

    Nobody volunteers "I feel exactly average" — the unprompted share only makes sense
    on a day that actually has a shape to it. Baseline PLUS drift, so a bland morning
    the day has since ground down becomes mentionable.

    Measured on `mood.energy`, NOT effective_energy(): the late-hour taper is a
    DELIVERY adjustment (don't claim to be wired at 1am), not a property of the day.
    Letting it feed this made the clock manufacture a reason to talk about himself —
    every mid-energy mood crossed the low bar after 8pm, taking the shipped pool from
    12 of 18 mentionable to nearly all of them, every evening.
    """
    mood = ensure_today(now)
    if mood is None:
        return False
    try:
        import config
        min_valence = abs(float(getattr(config, "REX_MOOD_SHARE_MIN_INTENSITY", 0.45)))
        low = float(getattr(config, "REX_MOOD_SHARE_LOW_ENERGY", 0.25))
        high = float(getattr(config, "REX_MOOD_SHARE_HIGH_ENERGY", 0.85))
    except Exception:
        min_valence, low, high = 0.45, 0.25, 0.85
    energy = mood.energy
    return abs(mood.valence) >= min_valence or energy <= low or energy >= high


def share_cue(now: Optional[datetime] = None) -> Optional[dict]:
    """The payload for VOLUNTEERING today's mood unprompted, or None.

    Returns None unless the mood is notable AND he hasn't already voiced it today —
    "spoken" is persisted, so telling Bret he's worn out at 9am means he won't
    announce it again at 2pm after a reboot. That is the same "don't repeat yourself
    at a person" instinct as greeting_cadence, applied to his own material.

    Deliberately does NOT roll the dice or check the relationship — the CALLER owns
    pacing and social fit (interaction._lean_mood_share_cue), the same split every
    other lull cue uses.
    """
    if not _enabled():
        return None
    mood = ensure_today(now)
    if mood is None or mood.spoken:
        return None
    if not is_notable(now):
        return None
    return {
        "label": mood.label,
        "line": mood.line,
        "because": mood.because,
        "shade": _shade(mood, effective_energy(now)),
        "seed_id": mood.seed_id,
    }


def prompt_section(now: Optional[datetime] = None) -> str:
    """The classic-prompt section (llm.assemble_system_prompt). Same content as
    prompt_lines with a heading, since the classic prompt is section-structured."""
    lines = prompt_lines(now)
    if not lines:
        return ""
    return "Rex's own state today:\n" + "\n".join(lines)


# ─────────────────────────────────────────────────────────────────────────────
# Public API — drift
# ─────────────────────────────────────────────────────────────────────────────

def note(kind: str, now: Optional[datetime] = None) -> Optional[DayMood]:
    """Nudge the mood off its baseline for a day event (see config.REX_MOOD_DRIFT).
    Unknown kinds are ignored. Cumulative drift is clamped on each axis so the day
    keeps its character — twenty good exchanges don't turn a flat day euphoric.

    RATE-LIMITED PER KIND (REX_MOOD_DRIFT_MIN_INTERVAL_SECS). Callers live on polling
    loops: `long_quiet` in particular is evaluated from the lull path, which can be
    consulted repeatedly inside one stretch of silence. Without pacing, a single quiet
    afternoon would drive the drift to its clamp within seconds and every subsequent
    event that day would be a no-op. Pacing here rather than at each call site means a
    future hook can't reintroduce the problem.
    """
    if not _enabled():
        return None
    kind = str(kind)
    dv, de = (_drift_table().get(kind) or (0.0, 0.0))[:2]
    if not dv and not de:
        return current()
    mood = ensure_today(now)
    if mood is None:
        return None
    interval = _drift_min_interval()
    stamp = time.monotonic()
    limit = _drift_limit()
    with _lock:
        last = _last_note_at.get(kind)
        if interval > 0 and last is not None and (stamp - last) < interval:
            return mood
        _last_note_at[kind] = stamp
        mood.drift_valence = _clamp(mood.drift_valence + float(dv), -limit, limit)
        mood.drift_energy = _clamp(mood.drift_energy + float(de), -limit, limit)
        mood.events.append(kind)
        if len(mood.events) > 40:
            del mood.events[:-40]
    _log.debug(
        "[rex_mood] %s → valence=%+.2f energy=%.2f", kind, mood.valence, mood.energy,
    )
    return mood


def note_spoken() -> None:
    """Record that he actually told someone how he is. This LOCKS the mood's framing:
    enrich() stops re-flavoring `because` afterward, because retconning the reason for
    a mood he already explained out loud is exactly the kind of thing people notice."""
    with _lock:
        if _current is not None:
            _current.spoken += 1


def _line_voices_mood(line: str, mood: DayMood) -> bool:
    """True when one of Rex's finished lines actually VOICED today's mood.

    Two ways to match: he used the mood word itself, or his line overlaps the authored
    example's distinctive words. Biased toward detecting — a false positive only stops
    a `because` from being upgraded, while a false negative lets him get retconned.
    """
    words = set(re.findall(r"[a-z']+", (line or "").lower()))
    if not words:
        return False
    # Label match on WHOLE TOKENS only. The first cut used a substring test against a
    # sorted word-soup, and "worn" inside "sworn" locked the spoken flag off a totally
    # unrelated line — which silently killed the unprompted share AND the enrich pass
    # for the rest of the day. Multi-word labels ("keyed-up") require every token.
    label_tokens = set((mood.label or "").lower().replace("-", " ").split())
    if label_tokens and label_tokens <= words:
        return True
    example = set(re.findall(r"[a-z']{5,}", (mood.line or "").lower())) - _STOPWORDS
    if len(example) < 3:
        return False
    return len(example & words) >= max(2, (len(example) + 2) // 3)


# Common-enough words that sharing them with the authored example proves nothing.
_STOPWORDS = {
    "about", "actually", "again", "along", "already", "always", "another", "anything",
    "around", "because", "before", "being", "better", "everything", "getting", "going",
    "honestly", "little", "might", "never", "nothing", "really", "should", "something",
    "still", "their", "there", "these", "thing", "things", "think", "those", "using",
    "usual", "which", "while", "would", "you're", "yours",
}


def note_spoken_if_voiced(line: str) -> bool:
    """Arm the spoken-lock if `line` voiced today's mood. PURE with respect to
    minting — never MINTS a mood (an idle line shouldn't create the day's mood as a
    side effect). Returns True on a match.

    Mirrors rex_pov.note_pov_spoken_if_voiced, which exists because the equivalent
    guard there was only ever armed from a dead code path and so never fired.
    """
    with _lock:
        mood = _current
    if mood is None or not (line or "").strip():
        return False
    if _line_voices_mood(line, mood):
        note_spoken()
        return True
    return False


def enrich(now: Optional[datetime] = None) -> bool:
    """Attach a real cause to a mood minted before the day's signals landed.

    Weather is ~1-3s behind boot and a cold news cache is ~45s behind it, so the mood
    minted at startup is usually a `chance` roll with no explanation. Once those land,
    upgrade the EXPLANATION — never the mood itself, which is his and already set.
    No-op once he has spoken it, or once a cause is already attached.
    Returns True when something was upgraded.
    """
    if not _enabled():
        return False
    mood = ensure_today(now)
    if mood is None:
        return False
    with _lock:
        if mood.spoken or mood.because:
            return False
    # allow_blocking=True: this runs on the background enrich thread, which can afford
    # the holiday feed's 5s-timeout first fetch that the boot-path mint must not pay.
    _tags, because, kind = _day_tags(now, allow_blocking=True)
    if not because:
        return False
    with _lock:
        if _current is None or _current.spoken or _current.because:
            return False
        _current.because = because
        _current.seed_kind = kind
    _log.info("[rex_mood] enriched %r with a %s cause: %s", mood.label, kind, because)
    return True


# ─────────────────────────────────────────────────────────────────────────────
# Persistence — resume today's mood across a reboot
# ─────────────────────────────────────────────────────────────────────────────

def _persist_enabled() -> bool:
    try:
        import config
        return bool(getattr(config, "REX_MOOD_PERSIST_ENABLED", True)) and _enabled()
    except Exception:
        return False


def _default_state_path() -> Path:
    return Path(__file__).resolve().parent.parent / "assets" / "memory" / "rex_mood_state.json"


def _state_path() -> Path:
    """Relative config overrides resolve against the PROJECT ROOT, not the cwd
    (current_events._path does this; rex_pov._state_path does not, and that's a bug
    waiting to bite whoever sets a relative path)."""
    try:
        import config
        raw = getattr(config, "REX_MOOD_STATE_PATH", None)
        if raw:
            p = Path(raw)
            if not p.is_absolute():
                p = Path(__file__).resolve().parent.parent / p
            return p
    except Exception:
        pass
    return _default_state_path()


def _under_test_runner() -> bool:
    """Keyed on the ENTRY POINT, never on `'unittest' in sys.modules` — the robot runs
    `python main.py`, and an incidental unittest import by any dependency would
    otherwise silently disable real persistence in production."""
    import os
    import sys
    if os.environ.get("DJR3X_MOOD_TEST_OPT_IN"):
        return False
    if os.environ.get("PYTEST_CURRENT_TEST"):
        return True
    argv0 = (sys.argv[0] if sys.argv else "").lower()
    return "unittest" in argv0 or "pytest" in argv0 or "py.test" in argv0


def _file_io_suppressed() -> bool:
    # A test that patched REX_MOOD_STATE_PATH to a temp file is exercising the writer
    # on purpose and is exempt. Both paths are built from the same expression so the
    # equality check can't silently fail and let the suite write the real file.
    return _under_test_runner() and _state_path() == _default_state_path()


def snapshot_state() -> Optional[dict]:
    """Serializable state: today's mood (with its accumulated drift) + the recent-mood
    memory that drives anti-repeat. Pure read; None when nothing is minted."""
    with _lock:
        if _current is None and not _recent_ids:
            return None
        mood = None
        if _current is not None:
            mood = {
                "date": _current.date,
                "seed_id": _current.seed_id,
                "because": _current.because,
                "seed_kind": _current.seed_kind,
                "minted_at": _current.minted_at,
                "drift_valence": round(_current.drift_valence, 4),
                "drift_energy": round(_current.drift_energy, 4),
                "events": list(_current.events),
                "spoken": int(_current.spoken),
            }
        return {"mood": mood, "recent": list(_recent_ids)}


def restore_state(data: Optional[dict], now: Optional[datetime] = None) -> bool:
    """Install a persisted snapshot. Returns True when TODAY's mood was restored.

    A snapshot from a previous day restores only the recent-mood memory (so
    anti-repeat still works) and leaves the mood unminted — waking up tomorrow means
    a new mood, which is the entire point. Label/valence/energy are re-read from the
    CURRENT seed pool by id, so editing a seed's wording takes effect immediately and
    a removed seed is dropped rather than resurrected from disk.
    """
    global _current, _recent_ids
    if not isinstance(data, dict):
        return False
    try:
        today = (now or datetime.now()).strftime("%Y-%m-%d")
        valid = {s["id"]: s for s in _seeds()}
        recent = [
            {"date": str(row.get("date")), "seed_id": str(row.get("seed_id"))}
            for row in (data.get("recent") or [])
            if isinstance(row, dict) and str(row.get("seed_id")) in valid
        ]
        keep = _recent_memory_days()
        if keep and len(recent) > keep:
            recent = recent[-keep:]

        blob = data.get("mood") if isinstance(data.get("mood"), dict) else None
        with _lock:
            _recent_ids = recent
            _current = None
            if not blob or str(blob.get("date")) != today:
                return False
            seed = valid.get(str(blob.get("seed_id")))
            if seed is None:
                return False
            limit = _drift_limit()
            _current = DayMood(
                date=today,
                seed_id=seed["id"],
                label=seed["label"],
                base_valence=seed["valence"],
                base_energy=seed["energy"],
                line=seed["line"],
                because=str(blob.get("because") or ""),
                seed_kind=str(blob.get("seed_kind") or "chance"),
                minted_at=str(blob.get("minted_at") or ""),
                drift_valence=_clamp(blob.get("drift_valence", 0.0), -limit, limit),
                drift_energy=_clamp(blob.get("drift_energy", 0.0), -limit, limit),
                events=[str(e) for e in (blob.get("events") or [])][-40:],
                spoken=int(blob.get("spoken") or 0),
            )
        _log.info(
            "[rex_mood] resumed %r for %s (drift %+.2f/%+.2f, spoken=%d)",
            _current.label, today, _current.drift_valence, _current.drift_energy,
            _current.spoken,
        )
        return True
    except Exception as exc:
        _log.debug("[rex_mood] restore_state failed: %s", exc)
        return False


def persist() -> None:
    """Write state to disk. Atomic (tmp + replace) because the mood is read on the
    next boot's greeting path and a torn read there would cost him his whole day."""
    if not _persist_enabled() or _file_io_suppressed():
        return
    try:
        state = snapshot_state()
        path = _state_path()
        if state is None:
            if path.exists():
                path.unlink()
            return
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp = path.with_suffix(".json.tmp")
        tmp.write_text(json.dumps(state, ensure_ascii=False, indent=2) + "\n",
                       encoding="utf-8")
        tmp.replace(path)
    except Exception as exc:
        _log.debug("[rex_mood] persist failed: %s", exc)


def load_persisted(now: Optional[datetime] = None) -> bool:
    """Read + install persisted state (call at startup, BEFORE the greeting path can
    read the mood). Returns True when TODAY's mood was resumed."""
    if not _persist_enabled() or _file_io_suppressed():
        return False
    try:
        path = _state_path()
        if not path.exists():
            return False
        return restore_state(json.loads(path.read_text(encoding="utf-8")), now=now)
    except Exception as exc:
        _log.debug("[rex_mood] load_persisted failed: %s", exc)
        return False


def clear() -> None:
    """Wipe in-memory mood state. NOT called from the session-reset bundle — a mood
    belongs to the DAY, not the conversation; clearing it when someone walks away
    would re-roll his personality every time the room empties. Tests use it."""
    global _current, _recent_ids, _last_note_at
    with _lock:
        _current = None
        _recent_ids = []
        _last_note_at = {}


# ─────────────────────────────────────────────────────────────────────────────
# Background enrichment
# ─────────────────────────────────────────────────────────────────────────────

_enrich_started = False


def start_background_enrich() -> None:
    """Fire-and-forget: re-check for a real cause once the async feeds land.

    Weather arrives 1-3s after chronoception starts and the daily news fetch can take
    ~45s, so the boot-time mint is usually causeless. Polls a few times, then gives up
    — a mood with no stated reason is perfectly fine, people have those.
    """
    global _enrich_started
    if _enrich_started or not _enabled():
        return
    _enrich_started = True

    def _loop() -> None:
        for delay in (5.0, 20.0, 60.0, 180.0):
            time.sleep(delay)
            try:
                if enrich():
                    return
            except Exception as exc:
                _log.debug("[rex_mood] enrich pass failed: %s", exc)

    threading.Thread(target=_loop, daemon=True, name="rex-mood-enrich").start()
