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
    # stemmed shards of function words ("this"→"thi", "didn't"→"didn") and
    # question filler — these matched every row and buried real topic hits
    "thi", "didn", "doesn", "isn", "wasn", "aren", "haven", "hasn", "wouldn",
    "couldn", "shouldn", "won", "ask", "went", "gone", "goes", "come", "came",
    "know", "remember", "recall", "hear", "heard",
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


def _dated_mentions(person_id: int, topic_tokens, limit: int = 4) -> list[str]:
    """The person's OWN dated words on the asked-about topic, from the persisted
    conversation_log — what makes "when did I mention going camping?" answerable
    with an actual date ('[2026-06-18] "I'm going camping next month"') instead
    of a vague 'this summer'. One line per day, newest first. [] on any failure."""
    if not topic_tokens:
        return []
    try:
        from memory import database as db, text_match
        rows = db.fetchall(
            """SELECT day, text FROM conversation_log
               WHERE person_id = ? ORDER BY ts DESC LIMIT 2000""",
            (int(person_id),),
        )
        hits: list[tuple[int, str, str]] = []   # (overlap, day, text)
        for r in rows:
            row = dict(r)
            raw = str(row.get("text") or "").strip()
            if raw.endswith("?"):
                continue   # their QUESTION about the topic isn't a mention of it
            overlap = text_match.overlap_count(raw, topic_tokens)
            if overlap <= 0:
                continue
            hits.append((overlap, str(row.get("day") or ""),
                         " ".join(raw.split())[:140]))
        # Strongest topic match first, then newest — a 2-token hit beats a fresh
        # 1-token graze; one line per day.
        hits.sort(key=lambda h: (h[0], h[1]), reverse=True)
        out: list[str] = []
        seen_days: set = set()
        for _overlap, day, text in hits:
            if day in seen_days:
                continue
            seen_days.add(day)
            out.append(f'[{day}] they said: "{text}"')
            if len(out) >= max(0, int(limit)):
                break
        return out
    except Exception as exc:
        _log.debug("[recall] dated mentions failed: %s", exc)
        return []


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
    mentions = _dated_mentions(int(person_id), tokens,
                               limit=int(_cfg("RECALL_MENTION_LIMIT", 4)))
    if not (facts or interests or qa or rels or episodes or mentions):
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
    if mentions:
        lines.append(
            "Their own dated words on this topic (use these DATES when they ask "
            "when something was said): " + " | ".join(mentions) + "."
        )
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


_NUM_WORDS = {
    "a": 1, "one": 1, "two": 2, "three": 3, "four": 4, "five": 5, "six": 6,
    "seven": 7, "eight": 8, "nine": 9, "ten": 10, "couple": 2, "few": 3,
}
_AGO_RE = re.compile(
    r"\b(a|one|two|three|four|five|six|seven|eight|nine|ten|couple|few|\d{1,2})"
    r"(?:\s+of)?\s+(day|week|month)s?\s+ago\b",
    re.IGNORECASE,
)


def parse_date_expression(text: str, today=None) -> Optional[tuple]:
    """Extract a (day_start, day_end, human_label) LOCAL date window from natural
    phrasing: 'on July 12 2026', 'yesterday', 'earlier today', 'last night',
    'this morning', 'last week', 'the other day', 'last time', 'two weeks ago',
    'three days ago'. None when absent. A bare month-day resolves to the most
    recent PAST occurrence; 'N weeks/months ago' returns a WINDOW (people are
    fuzzy about those)."""
    from datetime import date, timedelta
    t = " ".join(str(text or "").lower().split())
    if not t:
        return None
    today = today or date.today()

    m = _AGO_RE.search(t)
    if m:
        raw_n = m.group(1).lower()
        n = int(raw_n) if raw_n.isdigit() else _NUM_WORDS.get(raw_n, 1)
        unit = m.group(2).lower()
        if unit == "day":
            center, slack = n, (0 if n <= 2 else 1)
        elif unit == "week":
            center, slack = n * 7, 3
        else:  # month
            center, slack = n * 30, 7
        start = today - timedelta(days=center + slack)
        end = today - timedelta(days=max(1, center - slack))
        label = f"about {m.group(1)} {unit}{'s' if n != 1 else ''} ago"
        return (start.isoformat(), end.isoformat(), label)

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


_SENTENCE_SPLIT_RE = re.compile(r"(?<=[.!?…])\s+")


def _convo_recall_sentence(text: str) -> Optional[str]:
    """The sentence that asks what was talked about in a dated window — the
    conversation verb and the date expression must share ONE sentence. Field bug
    2026-08-01 23:42: "No, I didn't go paddleboarding YESTERDAY. …when did I TELL
    you about camping?" — cross-sentence matching grabbed yesterday's transcript
    and starved the camping memory block. Returns the matching sentence or None."""
    for sentence in _SENTENCE_SPLIT_RE.split(str(text or "")):
        if _CONVO_VERB_RE.search(sentence) and parse_date_expression(sentence):
            return sentence
    return None


def is_conversation_recall_question(text: str) -> bool:
    """True when the utterance asks what was talked about in a DATED window —
    a conversation verb plus a date expression in the SAME sentence."""
    return _convo_recall_sentence(text) is not None


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
    sentence = _convo_recall_sentence(utterance)
    if sentence is None:
        return []
    parsed = parse_date_expression(sentence)
    if parsed is None:
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
        "reading. Answer from THIS conversation ONLY: do not mix in things said "
        "on other days, and never present a recent or upcoming plan as if it was "
        "discussed back then. Never mention logs, records, or transcripts.",
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


# ── Statement-time known-context recall ─────────────────────────────────────────
# The inverse failure of memory_question_lines: rich recall fires when the person
# ASKS what Rex remembers, but when they STATE something he already holds a memory
# about, nothing was retrieved at all — so Rex heard about his own memories as news.
# Field 2026-08-03 18:53: Bret reported "I got all the new interns set up". The
# intern-training plan was person_event #13 (stored the night before; Rex even asked
# about it at 23:56) and a diary episode — yet the reply was "How many interns were
# there?", a stranger's question. A human spends what they know about the topic AT
# HAND; this block hands the reply model exactly that, with the one instruction that
# matters: connect, don't re-learn.

def _known_event_line(ev: dict) -> str:
    """One event rendered by lifecycle state, so the model knows whether it's reacting
    to an open plan (outcome unknown — maybe being reported RIGHT NOW), a done thing
    with a known outcome, or a canceled one it must not resurrect."""
    from memory import events as _events
    name = " ".join(str(ev.get("event_name") or "").split())
    when_told = _events.mentioned_when_label(ev.get("mentioned_at"))
    date_str = str(ev.get("event_date") or "").strip()[:10]
    dated = f" (was set for {date_str})" if date_str else ""
    outcome = " ".join(str(ev.get("outcome") or "").split())[:160]
    status = str(ev.get("status") or "planned").strip().lower()
    if status == "canceled":
        return (f"'{name}'{dated} — they CALLED IT OFF"
                + (f": \"{outcome}\"" if outcome else "") + ". Don't revive it.")
    if ev.get("followed_up") or status == "completed":
        return (f"'{name}'{dated} already happened; the outcome you were told: "
                + (f"\"{outcome}\"" if outcome else "(none recorded)") + ".")
    if status == "promised":
        return (f"something they promised to do ({when_told}): '{name}' — still open "
                f"as far as you know.")
    hedge = " It was tentative when they said it." if ev.get("hedged") else ""
    return (f"a plan they told you about {when_told}: '{name}'{dated}. STILL OPEN — "
            f"you have NOT heard how it went; if they're telling you now, react to "
            f"the outcome.{hedge}")


def known_context_lines(person_id: Optional[int], utterance: str) -> list[str]:
    """The known-context block: stored plans/events, diary episodes, and prior-session
    summaries that STRONGLY match what the person just SAID (not asked). [] when
    disabled, no known person, the utterance is too thin, it's a memory question
    (memory_question_lines owns those), or nothing matches. Conservative on purpose:
    a wrong 'you already know this' is worse than a missed connection, so matching
    uses text_match.strong_overlap, and there is no fuzzy fallback."""
    if person_id is None or not bool(_cfg("KNOWN_CONTEXT_RECALL_ENABLED", True)):
        return []
    text = " ".join(str(utterance or "").split())
    if len(text.split()) < int(_cfg("KNOWN_CONTEXT_MIN_WORDS", 3)):
        return []
    if is_memory_question(text):
        return []
    tokens = utterance_tokens(text)
    if not tokens:
        return []
    max_items = max(1, int(_cfg("KNOWN_CONTEXT_MAX_ITEMS", 3)))
    from memory import text_match

    items: list[str] = []

    # 1) Stored events/plans — the highest-value connection (they have lifecycle
    #    state the model must respect), so they claim slots first.
    try:
        from datetime import datetime, timedelta, timezone
        from memory import database as db
        lookback = int(_cfg("KNOWN_CONTEXT_EVENT_LOOKBACK_DAYS", 45))
        cutoff = (datetime.now(timezone.utc) - timedelta(days=lookback)).strftime(
            "%Y-%m-%dT%H:%M:%S"
        )
        date_cutoff = (datetime.now(timezone.utc) - timedelta(days=lookback)).strftime(
            "%Y-%m-%d"
        )
        rows = db.fetchall(
            """SELECT * FROM person_events
               WHERE person_id = ?
                 AND (mentioned_at >= ? OR (event_date IS NOT NULL AND event_date >= ?))
               ORDER BY mentioned_at DESC LIMIT 40""",
            (int(person_id), cutoff, date_cutoff),
        )
        seen_names: set = set()
        for r in rows:
            ev = dict(r)
            ev_text = " ".join([
                str(ev.get("event_name") or ""),
                str(ev.get("event_notes") or ""),
                str(ev.get("outcome") or ""),
            ])
            if not text_match.strong_overlap(tokens, ev_text):
                continue
            name_key = " ".join(str(ev.get("event_name") or "").lower().split())
            if not name_key or name_key in seen_names:
                continue
            seen_names.add(name_key)
            items.append(_known_event_line(ev))
            if len(items) >= max_items:
                break
    except Exception as exc:
        _log.debug("[recall] known-context events failed: %s", exc)

    # 2) Diary episodes — shared history in Rex's own words, dated.
    if len(items) < max_items:
        try:
            for row in search_episodes(tokens, person_id=int(person_id), limit=4):
                ep_text = f"{row.get('summary') or ''} {row.get('detail') or ''}"
                if not text_match.strong_overlap(tokens, ep_text):
                    continue
                items.append("from your own diary: " + _episode_line(row))
                if len(items) >= max_items:
                    break
        except Exception as exc:
            _log.debug("[recall] known-context episodes failed: %s", exc)

    # 3) Prior-session summaries — the distilled arc of past chats.
    if len(items) < max_items:
        try:
            from memory import conversations as conv_db
            for s in conv_db.get_conversation_history(int(person_id), limit=8):
                s_text = f"{s.get('summary') or ''} {s.get('topics') or ''}"
                if not text_match.strong_overlap(tokens, s_text):
                    continue
                date = str(s.get("session_date") or "")[:10]
                summary = " ".join(str(s.get("summary") or "").split())[:200]
                items.append(f"[{date}] a past chat of yours covered: {summary}")
                break   # one summary is plenty — events/episodes carry the specifics
        except Exception as exc:
            _log.debug("[recall] known-context summaries failed: %s", exc)

    if not items:
        return []
    lines = [
        "KNOWN CONTEXT — what they just said touches things you ALREADY know (below). "
        "React like someone who REMEMBERS: connect their words to the specifics you "
        "hold, and let any follow-up BUILD on them (they mention the race you knew "
        "they were training for → \"so did the knee hold up?\", never \"you ran a "
        "race?\"). Do NOT ask for anything already stated below, and do NOT react as "
        "if you're hearing about the topic for the first time. Never mention memory, "
        "records, or databases — you just remember."
    ]
    lines.extend("You know " + item for item in items)
    return lines
