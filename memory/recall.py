"""
memory/recall.py — query-time memory recall for the reply path.

Field failure (2026-08-01 22:38 session): Bret quizzed Rex on things the DBs had
held for weeks — favorite movie, job, hometown, dog, camping, "what movie did I
watch last night" (a literal rex_episodes row from the night before) — and Rex
denied knowing ANY of it. Root cause was retrieval, not storage: the lean reply
prompt carried only the top-4 facts + top-4 interests by STATIC score (junk
relationship edges won those slots), no topic awareness, and rex_episodes were
never read at question time at all (episodes only fed greetings/callbacks).

This module is the fix's core:

  * stems-of-the-utterance topic tokens for every reply turn, so the fact/interest
    ranking that already supports `topic_tokens` finally receives them;
  * `is_memory_question()` — detects "do you remember / what's my / did I tell
    you / what do you know about me" shapes;
  * `search_episodes()` — query-time episodic recall over rex_episodes (topic
    overlap required, salience+recency ranked, dated);
  * `memory_question_lines()` — the RICH recall block injected only on a memory
    question: identity facts as key:value, interests WITH their notes, direct
    Q&A answers, relationship edges, and matching diary episodes with dates —
    plus the instruction to answer FROM it and to admit a genuine blank honestly.

Fail-safe by construction: every public function returns an empty value on any
error so a broken DB can never break a reply.
"""

from __future__ import annotations

import logging
import re
from typing import Optional

_log = logging.getLogger(__name__)


def _cfg(name: str, default):
    try:
        import config
        return getattr(config, name, default)
    except Exception:
        return default


# ── Topic tokens for the current utterance ──────────────────────────────────────

# Question-machinery and filler stems that carry no topic signal — without this,
# "What movie did I WATCH LAST NIGHT?" matched every episode containing "did"
# (three 'I did an impersonation' rows outranked the actual movie memory).
_TOKEN_STOPSTEMS = {
    "did", "does", "was", "were", "are", "have", "has", "had", "will", "would",
    "could", "should", "can", "cant", "dont", "didnt", "you", "your", "yours",
    "the", "and", "for", "that", "this", "with", "what", "when", "where", "who",
    "whos", "how", "why", "which", "tell", "know", "say", "said", "told",
    "mention", "ever", "not", "about", "them", "they", "then", "than", "just",
    "like", "get", "got", "going", "gonna", "last", "night", "today", "yesterday",
    "tomorrow", "thing", "some", "any", "all", "one", "little", "bit", "also",
    "else", "okay", "yeah", "well", "really", "very", "much", "more",
}


def utterance_tokens(text: str) -> set:
    """Stemmed CONTENT tokens of what the person JUST said — the topic signal the
    per-silo rankers already accept, minus question-machinery words that would make
    everything match everything. Empty set on any failure."""
    try:
        from memory import text_match
        return {
            t for t in text_match.stems(text or "")
            if t not in _TOKEN_STOPSTEMS and text_match.stem(t) not in _TOKEN_STOPSTEMS
        }
    except Exception:
        return set()


# ── Memory-question detection ───────────────────────────────────────────────────
# Deliberately generous: a false positive only means a slightly bigger prompt for
# one turn; a false negative is the permanent-amnesia failure being fixed.
_MEMORY_QUESTION_RES = (
    re.compile(r"\bdo you (?:remember|recall|know)\b", re.IGNORECASE),
    re.compile(r"\bwhat(?:'s| is| are| was| were)? my\b", re.IGNORECASE),
    re.compile(r"\bwho(?:'s| is| was)? my\b", re.IGNORECASE),
    re.compile(r"\bwhen did i\b", re.IGNORECASE),
    re.compile(r"\bwhere did i\b", re.IGNORECASE),
    re.compile(r"\bwhat did i (?:say|tell|watch|do|mention)\b", re.IGNORECASE),
    re.compile(r"\bdid i (?:tell|mention|say|ever)\b", re.IGNORECASE),
    re.compile(r"\bhave i (?:ever\s+)?(?:told|mentioned|said)\b", re.IGNORECASE),
    re.compile(r"\bwhat (?:else\s+)?do you know about\b", re.IGNORECASE),
    re.compile(r"\btell me (?:.{0,20}\b)?about (?:me|myself|my)\b", re.IGNORECASE),
    re.compile(r"\bwhat movie did i\b", re.IGNORECASE),
    re.compile(r"\bwhat (?:do i do|i do) for\b", re.IGNORECASE),
    re.compile(r"\bwhere do i (?:live|work|come from)\b", re.IGNORECASE),
    re.compile(r"\byou (?:have no|don't have any) memory\b", re.IGNORECASE),
    re.compile(r"\bwhat are my\b", re.IGNORECASE),
    re.compile(r"\bwhat(?:'s| is) my favorite\b", re.IGNORECASE),
)


def is_memory_question(text: str) -> bool:
    """True when the utterance is asking Rex what he remembers/knows about the
    speaker — the cue to inject the rich recall block."""
    t = " ".join(str(text or "").split())
    if not t:
        return False
    return any(rx.search(t) for rx in _MEMORY_QUESTION_RES)


# ── Query-time episodic recall ──────────────────────────────────────────────────

def search_episodes(
    topic_tokens,
    person_id: Optional[int] = None,
    limit: int = 4,
    lookback_days: Optional[int] = None,
) -> list[dict]:
    """rex_episodes rows that MENTION the live topic, best-first. Requires at least
    one stemmed-token overlap (no topic tokens → nothing; this is query-time recall,
    not ambient reminiscing). Excludes scene captions. [] on any failure."""
    if not topic_tokens:
        return []
    try:
        from datetime import datetime, timedelta, timezone
        from memory import rex_db, text_match
        days = int(lookback_days or _cfg("RECALL_EPISODE_LOOKBACK_DAYS", 120))
        cutoff = (datetime.now(timezone.utc) - timedelta(days=days)).strftime(
            "%Y-%m-%d %H:%M:%S"
        )
        rows = rex_db.fetchall(
            "SELECT * FROM rex_episodes WHERE created_at >= ? AND kind != 'scene' "
            "ORDER BY created_at DESC LIMIT 400",
            (cutoff,),
        )
        scored: list[tuple[float, dict]] = []
        for r in rows:
            row = dict(r)
            text = f"{row.get('summary') or ''} {row.get('detail') or ''}"
            overlap = text_match.overlap_count(text, topic_tokens)
            if overlap <= 0:
                continue
            # A person-scoped hit outranks a general one; salience breaks ties.
            person_bonus = 0.5 if (
                person_id is not None and row.get("person_id") == person_id
            ) else 0.0
            try:
                salience = float(row.get("salience") or 0.5)
            except (TypeError, ValueError):
                salience = 0.5
            # Mild recency bias (like human memory): a half-life decay on a small
            # bonus, so among equal topic matches the FRESHER memory wins, but
            # recency can never outrank a stronger topic match (max +0.4 < 1 overlap).
            recency = 0.0
            try:
                ts = str(row.get("created_at") or "")[:19]
                age_days = max(0.0, (datetime.now(timezone.utc)
                                     - datetime.strptime(ts, "%Y-%m-%d %H:%M:%S")
                                     .replace(tzinfo=timezone.utc)).total_seconds() / 86400.0)
                halflife = float(_cfg("RECALL_EPISODE_RECENCY_HALFLIFE_DAYS", 21.0))
                recency = 0.4 * (0.5 ** (age_days / max(1.0, halflife)))
            except Exception:
                recency = 0.0
            scored.append((overlap + person_bonus + 0.25 * salience + recency, row))
        scored.sort(key=lambda pair: pair[0], reverse=True)
        # Collapse near-identical repeats (three 'I did an impersonation of Bret'
        # rows are ONE memory).
        seen: set = set()
        out: list[dict] = []
        for _, row in scored:
            key = re.sub(r"\s+", " ", str(row.get("summary") or "").strip().lower())
            if key in seen:
                continue
            seen.add(key)
            out.append(row)
            if len(out) >= max(0, int(limit)):
                break
        return out
    except Exception as exc:
        _log.debug("[recall] episode search failed: %s", exc)
        return []


def _episode_line(row: dict) -> str:
    date = str(row.get("created_at") or "")[:10]
    summary = " ".join(str(row.get("summary") or "").split())
    return f"[{date}] {summary}" if date else summary


# ── The rich recall block ───────────────────────────────────────────────────────

def _fact_pairs(person_id: int, topic_tokens, limit: int) -> list[str]:
    from memory import facts as facts_db
    out = []
    for f in facts_db.get_prompt_worthy_facts(
        person_id, limit=limit, topic_tokens=topic_tokens
    ):
        key = str(f.get("key") or "").replace("_", " ").strip()
        value = str(f.get("value") or "").strip()
        if key and value:
            hedge = " (inferred — hedge it)" if f.get("source") == "inferred" else ""
            out.append(f"{key}: {value}{hedge}")
    return out


def _interest_pairs(person_id: int, topic_tokens, limit: int) -> list[str]:
    from memory import interests as interests_db
    out = []
    for it in interests_db.get_interests_for_prompt(
        person_id, limit=limit, topic_tokens=topic_tokens
    ):
        name = str(it.get("name") or "").strip()
        if not name:
            continue
        notes = " ".join(str(it.get("notes") or "").split())
        out.append(f"{name} — {notes}" if notes else name)
    return out


def _qa_pairs(person_id: int, limit: int) -> list[str]:
    from memory import relationships as rel_db
    out = []
    for row in rel_db.get_qa_history(person_id)[-limit:]:
        row = dict(row)
        topic = str(row.get("question_key") or "").replace("_", " ").strip()
        answer = " ".join(str(row.get("answer_text") or "").split())
        if topic and answer:
            out.append(f"their answer about {topic}: \"{answer}\"")
    return out


def _relationship_lines(person_id: int) -> list[str]:
    from memory import social
    out = []
    for edge in social.get_outbound(person_id):
        label = str(edge.get("relationship") or "").strip()
        name = str(edge.get("to_name") or "").strip()
        if label and name:
            out.append(f"their {label}: {name}")
    return out


def memory_question_lines(person_id: Optional[int], utterance: str) -> list[str]:
    """The rich recall block for a direct memory question — [] when it isn't one,
    when there's no known person, or when recall is disabled. Each element is one
    system-prompt line (lean_brain renders them as '- ' bullets)."""
    if person_id is None or not bool(_cfg("RECALL_RICH_ENABLED", True)):
        return []
    if not is_memory_question(utterance):
        return []
    tokens = utterance_tokens(utterance)
    lines: list[str] = []
    try:
        facts = _fact_pairs(int(person_id), tokens,
                            int(_cfg("RECALL_RICH_FACT_LIMIT", 14)))
    except Exception as exc:
        _log.debug("[recall] facts failed: %s", exc)
        facts = []
    try:
        interests = _interest_pairs(int(person_id), tokens,
                                    int(_cfg("RECALL_RICH_INTEREST_LIMIT", 10)))
    except Exception as exc:
        _log.debug("[recall] interests failed: %s", exc)
        interests = []
    try:
        qa = _qa_pairs(int(person_id), int(_cfg("RECALL_RICH_QA_LIMIT", 8)))
    except Exception as exc:
        _log.debug("[recall] qa failed: %s", exc)
        qa = []
    try:
        rels = _relationship_lines(int(person_id))
    except Exception as exc:
        _log.debug("[recall] relationships failed: %s", exc)
        rels = []
    episodes = [
        _episode_line(r)
        for r in search_episodes(tokens, person_id=int(person_id),
                                 limit=int(_cfg("RECALL_EPISODE_LIMIT", 4)))
    ]
    if not (facts or interests or qa or rels or episodes):
        return []
    lines.append(
        "MEMORY QUESTION: they are asking what you REMEMBER. Everything below is "
        "your real memory of them — answer richly and specifically FROM it, in your "
        "own voice (never recite it as a list or mention databases/records). If the "
        "specific thing they asked about is genuinely NOT below, say so honestly and "
        "briefly — never invent a memory."
    )
    if facts:
        lines.append("Facts you know: " + "; ".join(facts) + ".")
    if rels:
        lines.append("People in their life: " + "; ".join(rels) + ".")
    if interests:
        lines.append("Their interests (with what you know about each): "
                     + " | ".join(interests) + ".")
    if qa:
        lines.append("Things they've told you directly: " + " | ".join(qa) + ".")
    if episodes:
        lines.append(
            "From your diary — dated things that actually happened (today is "
            "relative to these dates): " + " | ".join(episodes) + "."
        )
    return lines


# ── Date-targeted conversation recall ───────────────────────────────────────────
# Owner idea 2026-08-01: every spoken turn is persisted to conversation_log, so
# "what did we talk about on July 12?" / "earlier today?" can read the ACTUAL
# words back and let the one lean reply call summarize them in Rex's voice — no
# extra LLM call, no extra latency.

_MONTHS = {
    "january": 1, "february": 2, "march": 3, "april": 4, "may": 5, "june": 6,
    "july": 7, "august": 8, "september": 9, "october": 10, "november": 11,
    "december": 12,
}
_MONTH_DAY_RE = re.compile(
    r"\b(january|february|march|april|may|june|july|august|september|october|"
    r"november|december)\s+(\d{1,2})(?:st|nd|rd|th)?(?:,?\s*(\d{4}))?\b",
    re.IGNORECASE,
)
# Verbs/nouns that make a dated utterance a CONVERSATION-recall ask.
_CONVO_VERB_RE = re.compile(
    r"\b(talk(?:ed|ing)?|discuss(?:ed)?|chat(?:ted)?|conversation|"
    r"say(?:ing)?|said|tell|told|mention(?:ed)?|cover(?:ed)?)\b",
    re.IGNORECASE,
)


def parse_date_expression(text: str, today=None) -> Optional[tuple]:
    """Extract a (day_start, day_end, human_label) LOCAL date window from natural
    phrasing: 'on July 12 2026', 'yesterday', 'earlier today', 'last night',
    'this morning', 'last week', 'the other day', 'last time'. None when absent.
    A bare month-day resolves to the most recent PAST occurrence."""
    from datetime import date, timedelta
    t = " ".join(str(text or "").lower().split())
    if not t:
        return None
    today = today or date.today()

    m = _MONTH_DAY_RE.search(t)
    if m:
        month = _MONTHS[m.group(1).lower()]
        try:
            dom = int(m.group(2))
            year = int(m.group(3)) if m.group(3) else today.year
            d = date(year, month, dom)
        except ValueError:
            return None
        if not m.group(3) and d > today:
            d = date(today.year - 1, month, dom)   # bare "July 12" is never future
        iso = d.isoformat()
        return (iso, iso, d.strftime("%B %-d, %Y"))

    if re.search(r"\b(earlier today|today|this morning|this afternoon|this evening)\b", t):
        iso = today.isoformat()
        return (iso, iso, "earlier today")
    if re.search(r"\b(yesterday|last night)\b", t):
        iso = (today - timedelta(days=1)).isoformat()
        return (iso, iso, "yesterday")
    if re.search(r"\blast week\b", t):
        return ((today - timedelta(days=7)).isoformat(),
                (today - timedelta(days=1)).isoformat(), "last week")
    if re.search(r"\bthe other day\b", t):
        return ((today - timedelta(days=4)).isoformat(),
                (today - timedelta(days=1)).isoformat(), "the other day")
    if re.search(r"\blast time\b", t):
        return ("LAST_SESSION", "LAST_SESSION", "last time")
    return None


def is_conversation_recall_question(text: str) -> bool:
    """True when the utterance asks what was talked about in a DATED window —
    a conversation verb plus a parseable date expression."""
    t = str(text or "")
    return bool(_CONVO_VERB_RE.search(t)) and parse_date_expression(t) is not None


def _sample_turns(turns: list[dict], cap: int) -> list[dict]:
    """Evenly sample an over-long day down to `cap` turns so the block covers the
    WHOLE conversation, not just its tail."""
    if len(turns) <= cap:
        return turns
    step = len(turns) / float(cap)
    return [turns[int(i * step)] for i in range(cap)]


def conversation_recall_lines(person_id: Optional[int], utterance: str) -> list[str]:
    """The dated-conversation recall block: the actual logged turns from the asked-
    about window (plus any saved session summaries), with an instruction for the
    reply model to summarize them naturally. [] when the utterance isn't a dated
    conversation question, when logging is off, or when nothing was logged then."""
    if not bool(_cfg("CONVERSATION_LOG_ENABLED", True)):
        return []
    parsed = parse_date_expression(utterance)
    if parsed is None or not _CONVO_VERB_RE.search(str(utterance or "")):
        return []
    day_start, day_end, label = parsed
    try:
        from datetime import date
        from memory import conversations as conv_db
        if day_start == "LAST_SESSION":
            prev = conv_db.last_logged_day_before(date.today().isoformat())
            if not prev:
                return []
            day_start = day_end = prev
            label = f"last time ({prev})"
        turns = conv_db.get_logged_turns(day_start, day_end)
    except Exception as exc:
        _log.debug("[recall] conversation log read failed: %s", exc)
        return []
    if not turns:
        return [
            f"They're asking what you two talked about {label} — your log has "
            f"NOTHING from then. Say so honestly (maybe you weren't running, or "
            f"it was before your time); do not invent a conversation."
        ]
    # Trim leading Rex-only boot/filler lines — the conversation starts at the
    # first human turn (keeps "Please wait while I finish loading" out of every
    # recall).
    first_human = next(
        (i for i, t in enumerate(turns)
         if str(t.get("speaker") or "").strip().lower() not in ("rex", "dj-r3x", "djr3x")),
        None,
    )
    if first_human:
        turns = turns[first_human:]
    sampled = _sample_turns(turns, int(_cfg("RECALL_CONVO_MAX_TURNS", 40)))
    rendered = " | ".join(
        f"{str(t.get('speaker') or '?').split()[0]}: "
        + " ".join(str(t.get("text") or "").split())[:160]
        for t in sampled
    )
    lines = [
        f"CONVERSATION RECALL: they're asking what you two talked about {label}. "
        f"Below is the ACTUAL logged conversation from then"
        + (" (evenly sampled — it ran longer)" if len(turns) > len(sampled) else "")
        + ". Summarize it naturally in your own voice — the topics, the memorable "
        "beats, anything funny — like a friend recalling a chat, NOT a minutes "
        "reading. Never mention logs, records, or transcripts.",
        f"The conversation ({label}): {rendered}",
    ]
    # Session summaries for the window add the distilled arc when present.
    try:
        from memory import conversations as conv_db
        if person_id is not None:
            sums = [
                s for s in conv_db.get_conversation_history(int(person_id), limit=10)
                if day_start <= str(s.get("session_date") or "")[:10] <= day_end
                and str(s.get("summary") or "").strip()
            ]
            if sums:
                lines.append(
                    "Your own summaries of those session(s): "
                    + " | ".join(str(s["summary"]).strip() for s in sums[:3])
                )
    except Exception:
        pass
    return lines
