"""
awareness/current_events.py — Rex's daily "things I heard about" cache.

One web-search LLM call per DAY (hard-gated by the stored date) fetches the top
~5 notable/viral stories of the day/week via the OpenAI Responses API hosted
``web_search`` tool — the same plumbing intelligence/web_search.py uses for
reply-time lookups, because a knowledge-cutoff model asked "what's in the news"
without the tool would hallucinate last year's headlines.

The fetch runs in the BACKGROUND during startup model preloads (kicked from
main.py before the ready line) and only logs its haul; nothing is spoken at
startup. The stories are stored in a small JSON file
(config.CURRENT_EVENTS_PATH) shaped:

    {"date": "2026-07-17", "fetched_at": "...", "stories":
        [{"headline": str, "summary": str, "topic": str, "mentioned": bool}]}

CONSUMER: consciousness._step_news_remark surfaces ONE unmentioned story in a
mid-conversation lull as an invitation ("hey, did you hear about ...?") —
competing through the normal proactive-speech governor like every other
conversational avenue, never a scheduled broadcast. mark_mentioned() persists
the spent flag so a story is offered at most once across sessions.

Same-day restarts reuse the file (no second API call); a failed fetch leaves
the previous day's file in place — day-old stories beat none.
"""

from __future__ import annotations

import json
import logging
import re
import threading
from datetime import datetime
from pathlib import Path
from typing import Optional

import config

_log = logging.getLogger(__name__)

_lock = threading.Lock()
_fetch_started = False


def _path() -> Path:
    p = Path(getattr(config, "CURRENT_EVENTS_PATH", "assets/memory/current_events.json"))
    if not p.is_absolute():
        p = Path(__file__).resolve().parent.parent / p
    return p


def _today() -> str:
    return datetime.now().strftime("%Y-%m-%d")


def _load() -> dict:
    try:
        return json.loads(_path().read_text())
    except Exception:
        return {}


def _save(data: dict) -> None:
    try:
        p = _path()
        p.parent.mkdir(parents=True, exist_ok=True)
        tmp = p.with_suffix(".json.tmp")
        tmp.write_text(json.dumps(data, ensure_ascii=False, indent=2) + "\n")
        tmp.replace(p)                      # atomic — no torn reads, ever
    except Exception as exc:
        _log.warning("[current_events] save failed: %s", exc)


def is_fresh() -> bool:
    return _load().get("date") == _today()


# ─────────────────────────────────────────────────────────────────────────────
# Story timing — "did you hear about the eclipse TODAY" when it's next week
# ─────────────────────────────────────────────────────────────────────────────
#
# Field 2026-08-06 00:28: the stored story was "Total solar eclipse viewing push
# for August 12, 2026" and the summary said "August 12, 2026" twice — and Rex
# still opened with "did you hear about the eclipse today". A news frame implies
# immediacy, so a model handed a headline will reach for "today" unless the
# relative day is spelled out for it. Exactly the failure `_build_anticipation_
# prompt` already fixed for remembered events (a July-4 event opened as happening
# "tonight" on July 3), so it gets the same treatment: compute the delta HERE and
# state it, rather than trusting the model to subtract dates.

_MONTHS = {
    "jan": 1, "feb": 2, "mar": 3, "apr": 4, "may": 5, "jun": 6,
    "jul": 7, "aug": 8, "sep": 9, "oct": 10, "nov": 11, "dec": 12,
}
# EXACT month names — full or standard abbreviation. An earlier cut spelled these
# as `(jan|feb|...)[a-z]*`, which made any ordinary word starting with a month
# prefix into a month: "Officials **dec**lared 6 counties" parsed as December 6,
# "3 **sep**arate failures" as September 3, "4 **dec**ades" as December 4. On an
# otherwise undated story that REPLACED the safe "you don't know when" hedge with
# a confident, wrong date — strictly worse than the vagueness this fix removes.
_MONTH_RE = (
    r"(jan(?:uary)?|feb(?:ruary)?|mar(?:ch)?|apr(?:il)?|may|jun(?:e)?|jul(?:y)?"
    r"|aug(?:ust)?|sep(?:t|tember)?|oct(?:ober)?|nov(?:ember)?|dec(?:ember)?)"
)
_DATE_MDY = re.compile(rf"\b{_MONTH_RE}\.?\s+(\d{{1,2}})(?:st|nd|rd|th)?(?:,?\s+(\d{{4}}))?\b",
                       re.IGNORECASE)
# Day-first needs DATE-ISH CONTEXT — an ordinal suffix, an "of", or a trailing
# year. A bare number before a month name is usually a version or a count, not a
# day: "Pixel 9 August feature drop" is not August 9th, and "Season 3 September
# premiere" is not September 3rd.
_DATE_DMY_PATS = (
    re.compile(rf"\b(\d{{1,2}})(?:st|nd|rd|th)\s+(?:of\s+)?{_MONTH_RE}\b(?:,?\s+(\d{{4}}))?",
               re.IGNORECASE),
    re.compile(rf"\b(\d{{1,2}})\s+of\s+{_MONTH_RE}\b(?:,?\s+(\d{{4}}))?", re.IGNORECASE),
    re.compile(rf"\b(\d{{1,2}})\s+{_MONTH_RE},?\s+(\d{{4}})\b", re.IGNORECASE),
)
_DATE_ISO = re.compile(r"\b(\d{4})-(\d{2})-(\d{2})\b")


def _candidates(blob: str, today):
    """Every (date) candidate in `blob`, in the order found. Invalid calendar days
    are SKIPPED rather than ending the scan: an earlier cut returned on the first
    regex hit, so one bogus match ("Star Trek **mar**ks 60 years") suppressed a
    genuine date later in the same summary."""
    from datetime import date as _date
    out = []

    def _mk(y, m, d):
        try:
            return _date(int(y), int(m), int(d))
        except (ValueError, TypeError):
            return None

    def _month(tok):
        return _MONTHS.get(str(tok)[:3].lower())

    def _resolve(month, day, year):
        if not month:
            return None
        if year:
            return _mk(year, month, day)
        best = None
        for y in (today.year - 1, today.year, today.year + 1):
            cand = _mk(y, month, day)
            if cand and (best is None or abs((cand - today).days) < abs((best - today).days)):
                best = cand
        return best

    for m in _DATE_ISO.finditer(blob):
        d = _mk(m.group(1), m.group(2), m.group(3))
        if d:
            out.append((m.start(), d))
    for m in _DATE_MDY.finditer(blob):
        d = _resolve(_month(m.group(1)), m.group(2), m.group(3))
        if d:
            out.append((m.start(), d))
    for pat in _DATE_DMY_PATS:
        for m in pat.finditer(blob):
            d = _resolve(_month(m.group(2)), m.group(1), m.group(3))
            if d:
                out.append((m.start(), d))
    return [d for _, d in sorted(out, key=lambda pair: pair[0])]


def story_event_date(text: str, today=None):
    """First calendar date mentioned in `text` as a `date`, or None.

    A bare "August 12" (no year) resolves to whichever year puts it nearest today,
    so a story read in late December about "January 3" doesn't land eleven months
    in the past. Returns None rather than guessing when nothing parses.
    """
    from datetime import date as _date
    cands = _candidates(str(text or ""), today or _date.today())
    return cands[0] if cands else None


def story_event_dates(text: str, today=None) -> list:
    """EVERY distinct calendar date in `text`, ascending. Drives the multi-date
    branch of story_timing_clause — see there for why picking one would be worse."""
    from datetime import date as _date
    seen = _candidates(str(text or ""), today or _date.today())
    out = []
    for d in sorted(set(seen)):
        if d not in out:
            out.append(d)
    return out


def story_timing_clause(story, today=None) -> str:
    """A prompt clause pinning WHEN the story's event happens, or "".

    Handed to every path that speaks about a story so the relative day is stated
    rather than inferred. Empty when no date parses — better silent than wrong.
    """
    if not isinstance(story, dict):
        return ""
    if not bool(getattr(config, "NEWS_TIMING_CLAUSE_ENABLED", True)):
        return ""
    from datetime import date as _date
    today = today or _date.today()
    blob = f"{story.get('headline') or ''} {story.get('summary') or ''}"
    when = story_event_date(blob, today=today)
    if when is None:
        # No parseable date: still stop the reflexive "today", since a story can
        # be a day or two old by the time it is offered.
        return (
            "TIMING: the story gives no date, so you do NOT know when it happened "
            "or will happen. Do NOT say \"today\", \"tonight\", \"just now\", or "
            "imply it is happening as you speak."
        )
    # MULTIPLE dates ("Season 3 premiered July 23 and the finale airs August 9"):
    # the first one is not reliably the event being discussed, and asserting the
    # wrong date confidently is WORSE than the vague "today" this fix set out to
    # remove. List them and make the model pick, still banning the reflex.
    all_dates = story_event_dates(blob, today=today)
    if len(all_dates) > 1:
        parts = []
        for d in all_dates:
            delta = (d - today).days
            rel = ("today" if delta == 0 else "tomorrow" if delta == 1
                   else "yesterday" if delta == -1
                   else f"in {delta} days" if delta > 1 else f"{abs(delta)} days ago")
            parts.append(f"{d.strftime('%B %-d, %Y')} ({rel})")
        return (
            "TIMING: this story mentions more than one date — "
            + "; ".join(parts)
            + ". Work out which one the thing you are actually mentioning happens "
            "on, and phrase it accordingly. If you are not sure, do NOT state a "
            "day at all — and never call a future event \"today\" or \"tonight\"."
        )
    days = (when - today).days
    pretty = when.strftime("%B %-d, %Y")
    if days == 0:
        rel = "TODAY"
    elif days == 1:
        rel = "TOMORROW"
    elif days == -1:
        rel = "YESTERDAY"
    elif days > 1:
        rel = f"in {days} days — it has NOT happened yet"
    else:
        rel = f"{abs(days)} days ago — it is already past"
    return (
        f"TIMING: the event this story describes is on {pretty}, which is {rel}. "
        f"Phrase every time reference accordingly and never guess a different day; "
        f"in particular do NOT call a future event \"today\" or \"tonight\"."
    )


def stories() -> list:
    """Today's (or the most recent) story list; [] when never fetched."""
    return list(_load().get("stories") or [])


# ─────────────────────────────────────────────────────────────────────────────
# Fetch (once per day)
# ─────────────────────────────────────────────────────────────────────────────

_JSON_BLOCK_RE = re.compile(r"\[.*\]", re.S)


def _parse_stories(text: str) -> list:
    """Extract the JSON story array from the model's answer (tolerates markdown
    fences / prose around it). Returns [] when nothing parseable."""
    if not text:
        return []
    m = _JSON_BLOCK_RE.search(text)
    if not m:
        return []
    try:
        raw = json.loads(m.group(0))
    except Exception:
        return []
    out = []
    for item in raw if isinstance(raw, list) else []:
        if not isinstance(item, dict):
            continue
        headline = str(item.get("headline") or "").strip()
        summary = str(item.get("summary") or "").strip()
        if not headline or not summary:
            continue
        out.append({
            "headline": headline[:200],
            "summary": summary[:500],
            "topic": str(item.get("topic") or "").strip()[:40],
            "mentioned": False,
        })
    return out


def _fetch_via_web_search() -> list:
    """One Responses-API call with the hosted web_search tool. Raises on failure
    (the caller decides whether stale data is acceptable)."""
    from intelligence.web_search import _client, _search_model, strip_links  # shared plumbing

    n = int(getattr(config, "CURRENT_EVENTS_STORY_COUNT", 5))
    today_h = datetime.now().strftime("%B %d, %Y")
    prompt = (
        f"Today is {today_h}. Search the web (multiple searches if needed — e.g. "
        f'"top news {today_h}", "biggest stories this week", "viral news this week") '
        f"and return the {n} most notable or talked-about stories RIGHT NOW.\n\n"
        "Every story must be a CONCRETE EVENT — something that happened: a "
        "launch, a ruling, a record, a discovery, a release, a win, an outage. "
        "NEVER describe news outlets, homepages, calendars, roundup pages, or "
        '"coverage" itself — "BBC\'s homepage mixes headlines" is a FAILURE; if a '
        "search only surfaces meta-pages, search again with a different query. "
        "Mix hard news with lighter viral items; not five angles on one story. "
        "SPREAD the topics: at most ONE story about AI/tech companies or AI "
        "models — the rest must come from other spheres (world, science, sports, "
        "culture, weather, the genuinely weird). "
        "Skip paywalled minutiae, celebrity gossip about minors, and graphic "
        "tragedy. Return STRICT JSON only — an array of objects:\n"
        '[{"headline": "short headline", "summary": "1-2 plain sentences with '
        'the concrete facts", "topic": "one-or-two-word category"}]'
    )
    # Same model-resolution chain as reply-time web search (WEB_SEARCH_MODEL
    # override -> conversation model), falling back to the known tool-capable one.
    model = _search_model() or str(getattr(config, "WEB_SEARCH_FALLBACK_MODEL", "gpt-4o-mini"))
    resp = _client.responses.create(
        model=model,
        input=prompt,
        tools=[{"type": "web_search"}],
        max_output_tokens=int(getattr(config, "CURRENT_EVENTS_MAX_OUTPUT_TOKENS", 900)),
        timeout=float(getattr(config, "CURRENT_EVENTS_TIMEOUT_SECS", 45.0)),
    )
    text = (getattr(resp, "output_text", None) or "").strip()
    parsed = _parse_stories(text)
    for s in parsed:
        s["summary"] = strip_links(s["summary"])
        s["headline"] = strip_links(s["headline"])
    return parsed


def refresh_if_stale() -> bool:
    """Fetch today's stories unless the cache is already today's. Returns True
    when a fresh fetch was stored. Blocking (call from a background thread)."""
    if not bool(getattr(config, "CURRENT_EVENTS_ENABLED", True)):
        return False
    with _lock:
        if is_fresh():
            d = _load()
            _log.info("[current_events] cache is today's — no fetch (path=%s dated=%s "
                      "fetched_at=%s today=%s, %d stories).",
                      _path(), d.get("date"), d.get("fetched_at"), _today(),
                      len(d.get("stories") or []))
            return False
        try:
            from intelligence import connectivity
            if connectivity.is_offline():
                _log.info("[current_events] fetch skipped — offline mode")
                return False
        except ImportError:
            pass
        _log.info("[current_events] cache stale (dated=%s today=%s) — fetching.",
                  _load().get("date"), _today())
        try:
            fetched = _fetch_via_web_search()
        except Exception as exc:
            _log.warning("[current_events] fetch failed (%s) — keeping previous cache.", exc)
            return False
        if not fetched:
            _log.warning("[current_events] fetch returned no parseable stories — keeping previous cache.")
            return False
        _save({"date": _today(), "fetched_at": datetime.now().isoformat(timespec="seconds"),
               "stories": fetched})
        for s in fetched:
            _log.info("[current_events] %s: %s", s["topic"] or "news", s["headline"])
        return True


def start_background_refresh() -> None:
    """Fire-and-forget daily refresh (called from main.py during model preloads).
    Never blocks startup; at most one thread per process."""
    global _fetch_started
    if _fetch_started or not bool(getattr(config, "CURRENT_EVENTS_ENABLED", True)):
        return
    _fetch_started = True
    threading.Thread(target=refresh_if_stale, daemon=True,
                     name="current-events-refresh").start()


# ─────────────────────────────────────────────────────────────────────────────
# Consumer API (the lull remark step)
# ─────────────────────────────────────────────────────────────────────────────

def pick_story() -> Optional[dict]:
    """One not-yet-mentioned story, or None. Preserves fetch order (the model
    leads with the most notable). Refuses a cache older than
    CURRENT_EVENTS_MAX_AGE_HOURS — "did you hear" about stale news is worse
    than silence (field 2026-07-18: a yesterday-dated cache was consumed)."""
    d = _load()
    try:
        fetched = datetime.fromisoformat(str(d.get("fetched_at") or ""))
        age_h = (datetime.now() - fetched).total_seconds() / 3600.0
        if age_h > float(getattr(config, "CURRENT_EVENTS_MAX_AGE_HOURS", 36.0)):
            return None
    except Exception:
        return None
    for s in d.get("stories") or []:
        if not s.get("mentioned"):
            return dict(s)
    return None


def mark_mentioned(story: dict) -> None:
    """Persist the spent flag so a story is offered at most once, ever."""
    if not story:
        return
    with _lock:
        data = _load()
        changed = False
        for s in data.get("stories") or []:
            if s.get("headline") == story.get("headline") and not s.get("mentioned"):
                s["mentioned"] = True
                changed = True
                break
        if changed:
            _save(data)


# ─────────────────────────────────────────────────────────────────────────────
# Interest-tailored news (per-topic daily cache)
# ─────────────────────────────────────────────────────────────────────────────
# Rex knows people's interests (memory/interests.py). When a known person is
# engaged in conversation, their top interests get a per-TOPIC news fetch (one
# web-search call per topic per day, capped by INTEREST_NEWS_MAX_TOPICS_PER_DAY)
# so a lull can open with "seen the new Strange New Worlds episode?" instead of
# generic headlines. Topics are cached globally (not per person) so two people
# who both love volleyball share one fetch. Stored in the same JSON under
# "interest_news": {"<topic>": {"date": ..., "fetched_at": ..., "stories": [...]}}.

_interest_fetches_today: dict = {"date": None, "count": 0}
_interest_refresh_inflight: set = set()


def _norm_topic(topic: str) -> str:
    return re.sub(r"\s+", " ", str(topic or "").strip().lower())[:60]


def _fetch_interest_news_via_web_search(topic: str) -> list:
    """One Responses-API web-search call for recent news about ``topic``."""
    from intelligence.web_search import _client, _search_model, strip_links

    n = int(getattr(config, "INTEREST_NEWS_STORY_COUNT", 3))
    today_h = datetime.now().strftime("%B %d, %Y")
    prompt = (
        f"Today is {today_h}. Search the web for the most recent, notable news "
        f"about: {topic}. Return the {n} freshest CONCRETE items — a release, an "
        "episode, a match result, a discovery, an announcement, an event. Recency "
        "matters: prefer this week over this month. NEVER describe outlets, "
        "homepages, or coverage itself. Skip rumors and paywalled minutiae. "
        "Return STRICT JSON only — an array of objects:\n"
        '[{"headline": "short headline", "summary": "1-2 plain sentences with '
        'the concrete facts", "topic": "one-or-two-word category"}]'
    )
    model = _search_model() or str(getattr(config, "WEB_SEARCH_FALLBACK_MODEL", "gpt-4o-mini"))
    resp = _client.responses.create(
        model=model,
        input=prompt,
        tools=[{"type": "web_search"}],
        max_output_tokens=int(getattr(config, "INTEREST_NEWS_MAX_OUTPUT_TOKENS", 700)),
        timeout=float(getattr(config, "CURRENT_EVENTS_TIMEOUT_SECS", 45.0)),
    )
    text = (getattr(resp, "output_text", None) or "").strip()
    parsed = _parse_stories(text)
    for s in parsed:
        s["summary"] = strip_links(s["summary"])
        s["headline"] = strip_links(s["headline"])
    return parsed


def _interest_cache_fresh(entry: Optional[dict]) -> bool:
    return bool(entry) and entry.get("date") == _today()


def refresh_interest_news(topics: list) -> None:
    """Fetch today's news for each stale topic (blocking — call from a
    background thread). Respects the daily fetch budget; failures keep any
    previous day's entry (day-old beats none, same rule as the main cache)."""
    if not bool(getattr(config, "INTEREST_NEWS_ENABLED", True)):
        return
    try:
        from intelligence import connectivity
        if connectivity.is_offline():
            return
    except ImportError:
        pass
    budget = int(getattr(config, "INTEREST_NEWS_MAX_TOPICS_PER_DAY", 4))
    for raw_topic in topics:
        topic = _norm_topic(raw_topic)
        if not topic:
            continue
        with _lock:
            if _interest_fetches_today["date"] != _today():
                _interest_fetches_today.update(date=_today(), count=0)
            entry = (_load().get("interest_news") or {}).get(topic)
            if _interest_cache_fresh(entry):
                continue
            if _interest_fetches_today["count"] >= budget:
                _log.info("[interest_news] daily fetch budget (%d) spent — %r waits", budget, topic)
                return
            if topic in _interest_refresh_inflight:
                continue
            _interest_refresh_inflight.add(topic)
            _interest_fetches_today["count"] += 1
        try:
            fetched = _fetch_interest_news_via_web_search(topic)
        except Exception as exc:
            _log.warning("[interest_news] fetch failed for %r (%s) — keeping previous.", topic, exc)
            fetched = []
        finally:
            with _lock:
                _interest_refresh_inflight.discard(topic)
        if not fetched:
            continue
        with _lock:
            data = _load()
            data.setdefault("interest_news", {})[topic] = {
                "date": _today(),
                "fetched_at": datetime.now().isoformat(timespec="seconds"),
                "stories": fetched,
            }
            _save(data)
        for s in fetched:
            _log.info("[interest_news] %s: %s", topic, s["headline"])


def start_interest_refresh(topics: list) -> None:
    """Fire-and-forget background refresh for a person's interest topics.
    Cheap no-op when everything is fresh or the budget is spent."""
    clean = [_norm_topic(t) for t in (topics or []) if _norm_topic(t)]
    if not clean:
        return
    stale = []
    with _lock:
        cache = _load().get("interest_news") or {}
        for t in clean:
            if not _interest_cache_fresh(cache.get(t)) and t not in _interest_refresh_inflight:
                stale.append(t)
    if not stale:
        return
    threading.Thread(
        target=refresh_interest_news, args=(stale,), daemon=True,
        name="interest-news-refresh",
    ).start()


def pick_interest_story(topics: list) -> Optional[tuple]:
    """First not-yet-mentioned story across the given topics (today's or
    yesterday's entry, same freshness bar as pick_story). Returns
    (topic, story) or None."""
    d = _load()
    cache = d.get("interest_news") or {}
    max_age_h = float(getattr(config, "CURRENT_EVENTS_MAX_AGE_HOURS", 36.0))
    for raw_topic in topics:
        topic = _norm_topic(raw_topic)
        entry = cache.get(topic)
        if not entry:
            continue
        try:
            fetched = datetime.fromisoformat(str(entry.get("fetched_at") or ""))
            if (datetime.now() - fetched).total_seconds() / 3600.0 > max_age_h:
                continue
        except Exception:
            continue
        for s in entry.get("stories") or []:
            if not s.get("mentioned"):
                return topic, dict(s)
    return None


def mark_interest_story_mentioned(topic: str, story: dict) -> None:
    """Persist the spent flag for an interest story (offered at most once)."""
    if not story:
        return
    topic = _norm_topic(topic)
    with _lock:
        data = _load()
        entry = (data.get("interest_news") or {}).get(topic) or {}
        changed = False
        for s in entry.get("stories") or []:
            if s.get("headline") == story.get("headline") and not s.get("mentioned"):
                s["mentioned"] = True
                changed = True
                break
        if changed:
            _save(data)
