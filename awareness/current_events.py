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
