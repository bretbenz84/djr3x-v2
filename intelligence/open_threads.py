"""
intelligence/open_threads.py — cross-session "did you ever...?" follow-ups.

The diary extractor (llm.generate_diary_entry, 2026-07-17 rework) stores
open_threads on each conversation_summary episode — the things a person left
unresolved ("whether the dentist appointment happened"). This module is the
CONSUMER: when that person is back and the conversation lulls, Rex asks about
ONE of them. This is the single feature most responsible for "whoa, he
remembered" — everything else in the memory stack feeds it.

Rules:
  * Freshness window: threads younger than OPEN_THREAD_MIN_AGE_HOURS feel
    like Rex forgot you just told him; older than OPEN_THREAD_MAX_AGE_DAYS
    feel like surveillance. Both configurable.
  * A thread is asked AT MOST ONCE, ever: spending rewrites the episode's
    detail JSON (threads_asked) so it survives restarts.
  * Surfacing runs through consciousness (_step_open_thread_followup) at a
    priority ABOVE lull callbacks and news — a personal follow-through beats
    banked humor and headlines.

Fail-safe throughout; reads/writes rex.db via memory.rex_db.
"""

from __future__ import annotations

import json
import logging
import re
from datetime import datetime, timedelta, timezone
from typing import Optional

import config
from memory import rex_db

_log = logging.getLogger(__name__)

# Bookkeeping shapes — maintenance of Rex's own records (name corrections,
# mishearings, forget/privacy requests, complaints about his hearing/memory).
# Never real threads: a wrong-name fix became "did JT's name change settle in
# okay?" and a guest's "I'm not into the data tracking thing" became "did JT
# ever make peace with the data tracking?" (field 2026-08-02/03). Lives here —
# the CONSUMER — so threads already stored before the write-side filter shipped
# die at read time too; the diary extractor (intelligence/llm.py) imports this
# same pattern for its write-side guarantee.
BOOKKEEPING_RE = re.compile(
    r"\bname\s+(?:change|correction|update|swap|mix-?up|situation)\b|"
    r"\b(?:right|real|correct|wrong|new)\s+name\b|"
    r"\bwhat\s+to\s+call\s+(?:him|her|them|me)\b|"
    r"\bmis(?:heard|hearing|hears?)\b|\bmis-?transcri|"
    r"\b(?:asked?|wants?)\s+(?:rex\s+|you\s+|me\s+)?to\s+forget\b|"
    r"\bforget\s+(?:him|her|them|me|his|their)\b|"
    r"\bdata[\s-]?track|\btracking\s+thing\b|\bbeing\s+tracked\b|"
    r"\bprivacy\s+(?:request|concern|thing)\b|"
    r"\b(?:rex|your|my)\s+(?:memory\s+banks?|hearing|transcription|audio|"
    r"circuits?|systems?|program(?:ming)?)\b",
    re.IGNORECASE,
)

# Getting-to-know-you curiosity shapes — the diary model files these as
# "threads" after any personal revelation, but they are QUESTIONS REX WANTS TO
# ASK, not things the person left unresolved. Spoken later as "you mentioned
# X", they become false memories (field 2026-08-02 13:03). Canonical here since
# 2026-09-03; the diary extractor (intelligence/llm.py) imports it for the
# write side and pending_for_person kills already-stored ones.
#   * favourites / preferences / opinions: "whether Bret has any favorite
#     quotes from Clinton" (field 2026-09-03 11:45 — the whole session was
#     "Impersonate Bill Clinton", a bit Rex performed; the next morning's lull
#     opened with "The Clinton quotes came up the other day — did you ever pick
#     one?", a follow-up on nothing).
#   * performance requests to Rex (impersonations, impressions, voices, jokes,
#     songs): the bit ended when Rex did it.
CURIOSITY_RE = re.compile(
    r"\bhow\b[^,;.]*\bmet\b|"
    r"\bwhat\b[^,;.]*\b(?:enjoy|like)s?\b[^,;.]*\btogether\b|"
    r"\bwhat\b[^,;.]*\bdo(?:es)?\s+together\b|"
    r"\bwhat activities\b|"
    r"\bif\b[^,;.]*\bhas (?:any )?plans\b|"
    r"\bwhere\b[^,;.]*\b(?:is |are )?from\b|"
    r"\bfavou?rite\b|\bprefer(?:s|red|ence)?\b|\bopinions?\s+(?:on|of|about)\b|"
    r"\b(?:any|which|what)\s+(?:other\s+)?(?:quotes?|lines?|sayings?)\b|"
    r"\bimpersonat|\bimpressions?\s+of\b|\bdo\s+(?:a|an|the|another)\s+(?:voice|accent|impression|impersonation)\b|"
    r"\b(?:another|more|other)\s+(?:impersonations?|impressions?|voices?|jokes?|songs?)\b",
    re.IGNORECASE,
)

# Game-table mechanics — chatter from a game REX HIMSELF hosted (Jeopardy,
# trivia): score disputes, board requests, whose turn, how the game proceeds.
# Rex administers the game, so these resolve inside it and die with it — never
# life events (field 2026-08-26: "take her points away, she cheated" became the
# next-day cold open "T'Joy's points came up the other day — did they actually
# get taken away?"). Same dual-guard shape as BOOKKEEPING_RE: the diary
# extractor (intelligence/llm.py) imports this for its write-side guarantee,
# and pending_for_person kills threads stored before it shipped. The game
# HAVING happened (who played, who won) is still memorable — that belongs in
# the diary NOTE, not in threads.
GAME_MECHANICS_RE = re.compile(
    r"\bpoints?\b.{0,40}\b(?:taken|took|deduct\w*|removed?|restored?|awarded|given)\b|"
    r"\b(?:take|taking|took|giv(?:e|ing)|got)\b.{0,20}\bpoints?\s+(?:away|back)\b|"
    r"\bcheat\w*\b.{0,40}\b(?:game|points?|jeopardy|trivia|score\w*)\b|"
    r"\b(?:game|jeopardy|trivia|score\w*|points?)\b.{0,40}\bcheat\w*\b|"
    r"\b(?:game|board|round)\b.{0,30}\b(?:proceed|continue|resume|restart)\w*\b|"
    r"\bhow\s+the\s+game\b|\bwhose\s+turn\b|"
    r"\bnext\s+(?:clue|category|square)\b|\b(?:pick|choose|chos\w+)\w*\b.{0,15}\bcategory\b|"
    r"\bdaily\s+double\b|\bfinal\s+jeopardy\b|\bdollar\s+value\b|"
    r"\bscore\w*\b.{0,30}\b(?:settl\w+|correct\w+|adjust\w+|fix\w+|final)\b",
    re.IGNORECASE,
)


def _age_days(created_at: str) -> Optional[float]:
    try:
        dt = datetime.strptime(created_at, "%Y-%m-%d %H:%M:%S")
        return (datetime.now() - dt).total_seconds() / 86400.0
    except Exception:
        return None


def _resolved_event_token_sets(person_id: int) -> list:
    """Content-token sets of this person's recently RESOLVED/canceled plans
    (person_events). An episode open thread about the same plan must not re-ask
    what a follow-up already settled — field 2026-08-19 20:01: 48 seconds after
    "No, I didn't go" resolved the library follow-up, this lane asked "The other
    day you mentioned the Obama library — did that actually happen?"."""
    try:
        from memory import database as people_db
        from memory import dedup
        guard_days = float(getattr(config, "OPEN_THREAD_RESOLVED_EVENT_GUARD_DAYS", 14.0))
        cutoff = (datetime.now(timezone.utc) - timedelta(days=guard_days)).isoformat()
        rows = people_db.fetchall(
            """SELECT event_name FROM person_events
               WHERE person_id = ?
                 AND (followed_up = TRUE
                      OR COALESCE(status, 'planned') IN ('completed', 'canceled'))
                 AND updated_at >= ?""",
            (int(person_id), cutoff),
        )
        out = []
        for r in rows:
            toks = dedup.event_content_tokens(str(r["event_name"] or ""))
            if len(toks) >= 2:      # one shared token must never nuke a thread
                out.append(toks)
        return out
    except Exception:
        return []


def _thread_covers_resolved_plan(thread: str, resolved: list) -> bool:
    if not resolved:
        return False
    try:
        from memory import dedup
        thread_tokens = set(dedup._token_set(thread))
    except Exception:
        return False
    return any(toks <= thread_tokens for toks in resolved)


# Words a diary thread uses to FRAME its subject ("whether the circus happened",
# "how the dentist appointment went") — not the subject itself. Stripped before the
# session-awareness check below so the match is on what the thread is ABOUT.
_THREAD_FRAME_WORDS = (
    "whether if how when what did does ever still yet actually really happened "
    "happen happens went goes ended end turned turn out up okay ok well later "
    "next again after before around eventually finally"
)
_thread_frame_tokens: Optional[frozenset] = None


def _frame_tokens() -> frozenset:
    global _thread_frame_tokens
    if _thread_frame_tokens is None:
        try:
            from memory import dedup
            _thread_frame_tokens = frozenset(dedup._tokens(_THREAD_FRAME_WORDS))
        except Exception:
            _thread_frame_tokens = frozenset()
    return _thread_frame_tokens


def _session_transcript_tokens() -> frozenset:
    """Content tokens of everything said THIS session (both sides), for the
    session-awareness guard. Empty when the transcript is unavailable."""
    try:
        from memory import conversations as conv_memory
        from memory import dedup
        toks: set = set()
        for turn in conv_memory.get_session_transcript() or []:
            toks.update(dedup._token_set(str((turn or {}).get("text") or "")))
        return frozenset(toks)
    except Exception:
        return frozenset()


def _thread_discussed_this_session(thread: str, session_tokens) -> bool:
    """True when the thread's SUBJECT already came up in this session's transcript.

    Field 2026-09-01 23:05:13: the reply model had already asked "did the circus
    actually happen?" at 23:01:49 (the previous session's recap sits in its prompt)
    and Bret had answered at length — then the lull lane, which only knows about
    threads IT spent, asked "The circus came up the other day — did you end up
    enjoying it?" and Bret had to say "we just talked about that". Only the lull
    lane marks threads asked; the reply model raising the same subject from the
    recap is invisible to it. So a thread whose subject the room has already
    covered this session is held back (it stays unspent — a later session can still
    raise it if the diary never closed it).

    Match rule: at least half of the thread's content tokens (frame words like
    "whether"/"happened" stripped; minimum one) appear in the session transcript.
    A one-word core ("circus") matches on that word; a two-word core ("dentist
    appointment") needs one of them; a false hold only delays a bring-up, while a
    false miss is the duplicate question the owner complained about."""
    if not session_tokens:
        return False
    try:
        from memory import dedup
        core = {t for t in dedup.event_content_tokens(thread) if t not in _frame_tokens()}
    except Exception:
        return False
    if not core:
        return False
    hit = len(core & set(session_tokens))
    need = max(1, (len(core) + 1) // 2)
    return hit >= need


def _upcoming_plan_hold(person_id: int) -> "tuple[list, bool]":
    """(content-token sets of dated FUTURE plans, any of them travel-shaped) —
    an episode thread about that plan ("how long they will stay in Huntsville",
    field 2026-09-03 12:17, five days before the owner leaves) must wait until
    the date has passed. The plan rows are people.db person_events."""
    try:
        from memory import events as events_mod
        from memory import dedup
        upcoming = events_mod.get_upcoming_events(int(person_id)) or []
    except Exception:
        return [], False
    horizon = float(getattr(config, "OPEN_THREAD_UPCOMING_PLAN_HOLD_DAYS", 21.0))
    limit = (datetime.now() + timedelta(days=horizon)).date().isoformat()
    toks, travel = [], False
    for plan in upcoming:
        if str(plan.get("event_date") or "") > limit:
            continue
        name = str(plan.get("event_name") or "")
        t = dedup.event_content_tokens(name)
        if t:
            toks.append(frozenset(t))
        travel = travel or events_mod.looks_like_travel(name)
    return toks, travel


def _thread_about_upcoming_plan(thread: str, plan_tokens: list, travel: bool) -> bool:
    try:
        from memory import dedup, events as events_mod
        mine = dedup.event_content_tokens(thread)
    except Exception:
        return False
    if any(mine & p for p in plan_tokens):
        return True
    return bool(travel and events_mod.looks_like_travel(thread))


def pending_for_person(person_id: int) -> list:
    """Unasked open threads for this person, freshest-first:
    [{"episode_id", "thread", "age_days"}]. Empty when none qualify."""
    if person_id is None:
        return []
    min_age_d = float(getattr(config, "OPEN_THREAD_MIN_AGE_HOURS", 6.0)) / 24.0
    max_age_d = float(getattr(config, "OPEN_THREAD_MAX_AGE_DAYS", 21.0))
    resolved = _resolved_event_token_sets(person_id)
    session_tokens = _session_transcript_tokens()
    plan_tokens, plan_travel = _upcoming_plan_hold(person_id)
    try:
        rows = rex_db.fetchall(
            "SELECT id, created_at, detail FROM rex_episodes "
            "WHERE kind = 'conversation_summary' AND person_id = ? "
            "ORDER BY created_at DESC LIMIT 40",
            (int(person_id),),
        )
    except Exception:
        return []
    out = []
    for row in rows:
        age = _age_days(str(row["created_at"] or ""))
        if age is None or age < min_age_d or age > max_age_d:
            continue
        try:
            detail = json.loads(row["detail"] or "{}")
        except Exception:
            continue
        asked = {str(t).strip().lower() for t in (detail.get("threads_asked") or [])}
        for thread in detail.get("open_threads") or []:
            t = str(thread).strip()
            if not t or t.lower() in asked:
                continue
            if BOOKKEEPING_RE.search(t):
                _log.info(
                    "[open_threads] dropping stored thread %r — bookkeeping "
                    "about Rex's own records, not a life event", t,
                )
                continue
            if GAME_MECHANICS_RE.search(t):
                _log.info(
                    "[open_threads] dropping stored thread %r — game-table "
                    "mechanics from a game Rex hosted, not a life event", t,
                )
                continue
            if CURIOSITY_RE.search(t):
                _log.info(
                    "[open_threads] dropping stored thread %r — curiosity / "
                    "preference / performance-bit shape, not something they "
                    "left unresolved", t,
                )
                continue
            if _thread_covers_resolved_plan(t, resolved):
                _log.info(
                    "[open_threads] dropping stored thread %r — its plan was "
                    "already resolved by a follow-up", t,
                )
                continue
            if (plan_tokens or plan_travel) and _thread_about_upcoming_plan(t, plan_tokens, plan_travel):
                # Held, not spent: it is about a trip that has not happened yet.
                _log.info(
                    "[open_threads] holding thread %r — it is about a plan that "
                    "is still ahead of them", t,
                )
                continue
            if _thread_discussed_this_session(t, session_tokens):
                # Held, not spent: the thread stays unasked for a later session.
                _log.info(
                    "[open_threads] holding thread %r — its subject already came "
                    "up this session", t,
                )
                continue
            out.append({"episode_id": int(row["id"]), "thread": t, "age_days": age})
    return out


def mark_asked(episode_id: int, thread: str) -> None:
    """Spend a thread permanently (persisted into the episode's detail JSON)."""
    try:
        row = rex_db.fetchone("SELECT detail FROM rex_episodes WHERE id = ?", (int(episode_id),))
        if row is None:
            return
        try:
            detail = json.loads(row["detail"] or "{}")
        except Exception:
            detail = {}
        asked = list(detail.get("threads_asked") or [])
        if thread not in asked:
            asked.append(thread)
        detail["threads_asked"] = asked
        rex_db.execute(
            "UPDATE rex_episodes SET detail = ? WHERE id = ?",
            (json.dumps(detail, ensure_ascii=False)[:4000], int(episode_id)),
        )
    except Exception as exc:
        _log.debug("[open_threads] mark_asked failed: %s", exc)


def describe_age(age_days: float) -> str:
    """Human phrasing for how long ago the thread was left open."""
    if age_days < 1.0:
        return "earlier today"
    if age_days < 2.0:
        return "yesterday"
    if age_days < 8.0:
        return f"{int(round(age_days))} days ago"
    return "a while back"
