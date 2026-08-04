"""
memory/boundaries.py - conversational boundaries and preferences.

These are not factual biography. They are consent/preferences for how Rex should
talk with a person: topics not to ask about, jokes not to make, and appearance
or check-in areas to avoid unless the person reopens them.
"""

from __future__ import annotations

import logging
import re
from datetime import datetime, timezone
from typing import Optional

from memory import database as db

_log = logging.getLogger(__name__)

_DEFAULT_TOPIC = "current topic"

_TOPIC_ALIASES = {
    "all": "anything",
    "anything": "anything",
    "everything": "anything",
    "questions": "questions",
    "question": "questions",
    "personal questions": "questions",
    "that": _DEFAULT_TOPIC,
    "this": _DEFAULT_TOPIC,
    "it": _DEFAULT_TOPIC,
    "me": "anything",
    "how i am doing": "how are you",
    "how i'm doing": "how are you",
    "how im doing": "how are you",
    "how are you": "how are you",
    "how i'm feeling": "how are you",
    "how i feel": "how are you",
    "my appearance": "appearance",
    "appearance": "appearance",
    "my face": "face",
    "face": "face",
    "my voice": "voice",
    "voice": "voice",
    "identity": "identity",
    "name": "identity",
    "my body": "body",
    "body": "body",
    "my weight": "body",
    "weight": "body",
    "work": "work",
    "my job": "work",
    "job": "work",
    "shirt": "clothing",
    "my shirt": "clothing",
    "clothes": "clothing",
    "clothing": "clothing",
}

_BOUNDARY_PATTERNS: list[tuple[str, re.Pattern[str]]] = [
    ("roast", re.compile(
        r"\b(?:don'?t|do not|stop|please don'?t|please do not)\s+"
        r"(?:roast|tease|make fun of|joke about)\s+"
        r"(?:me\s+)?(?:about|for|over)?\s*(?P<topic>[^.?!,;]+)",
        re.IGNORECASE,
    )),
    ("ask", re.compile(
        r"\b(?:don'?t|do not|stop|please don'?t|please do not)\s+"
        r"(?:ask|question|bring up)\s+"
        r"(?:me\s+)?(?:about|on)?\s*(?P<topic>[^.?!,;]+)",
        re.IGNORECASE,
    )),
    ("mention", re.compile(
        r"\b(?:don'?t|do not|stop|please don'?t|please do not)\s+"
        r"(?:mention|comment on|talk about|bring up)\s+"
        r"(?:my\s+|the\s+)?(?P<topic>[^.?!,;]+)",
        re.IGNORECASE,
    )),
    # Softened stand-downs (field 2026-08-03 20:03: "we don't need to bring up the
    # website anymore" matched NOTHING — every pattern above expects a bare
    # imperative — so the turn fell through to the plain reply LLM, which said
    # "Understood" and then kept probing the same topic). First-person-plural /
    # second-person polite forms with a NAMED topic; "anymore" and "no need" carry
    # the same durable weight as an imperative. The lookahead keeps pronoun objects
    # out of the topic slot — those belong to the generic patterns + fallback.
    ("mention", re.compile(
        r"\b(?:we|you)\s+(?:don'?t|do not)\s+(?:need|have)\s+to\s+"
        r"(?:bring up|talk about|mention|discuss|keep (?:bringing up|talking about|mentioning))\s+"
        r"(?:my\s+|the\s+)?(?P<topic>(?!(?:it|that|this)\b)[^.?!,;]+)",
        re.IGNORECASE,
    )),
    ("mention", re.compile(
        r"\b(?:(?:there'?s\s+)?no need to|you can stop|we can stop|let'?s stop|"
        r"let'?s not|quit)\s+"
        r"(?:bringing up|talking about|mentioning|asking about|discussing|"
        r"bring up|talk about|mention|ask about|discuss)\s+"
        r"(?:my\s+|the\s+)?(?P<topic>(?!(?:it|that|this)\b)[^.?!,;]+)",
        re.IGNORECASE,
    )),
    ("ask", re.compile(
        r"\b(?:i hate|i don'?t like|i do not like)\s+"
        r"(?:being\s+)?(?:asked|getting asked)\s+"
        r"(?:about\s+)?(?P<topic>[^.?!,;]+)",
        re.IGNORECASE,
    )),
    ("ask", re.compile(
        r"\b(?:no more|stop with the|enough with the)\s+"
        r"(?P<topic>questions?|personal questions?)\b",
        re.IGNORECASE,
    )),
]

_GENERIC_BOUNDARY_PATTERNS: list[tuple[str, re.Pattern[str]]] = [
    ("roast", re.compile(
        r"\b(?:don'?t|do not|stop|please don'?t|please do not)\s+"
        r"(?:roast|tease|make fun of|joke about)\s+me\b",
        re.IGNORECASE,
    )),
    ("mention", re.compile(
        r"\b(?:drop it|leave it alone|let'?s leave it|"
        r"stop talking about (?:that|this|it)|"
        r"don'?t talk about (?:that|this|it)(?: anymore| again)?|"
        r"do not talk about (?:that|this|it)(?: anymore| again)?|"
        r"don'?t bring (?:that|this|it) up(?: anymore| again)?|"
        r"do not bring (?:that|this|it) up(?: anymore| again)?|"
        # softened pronoun stand-downs — "we don't need to bring it up (anymore)"
        r"(?:we|you) (?:don'?t|do not) (?:need|have) to "
        r"(?:bring (?:that|this|it) up|talk about (?:that|this|it)|"
        r"mention (?:that|this|it))(?: anymore| again)?|"
        r"(?:there'?s )?no need to (?:bring (?:that|this|it) up|"
        r"talk about (?:that|this|it)|mention (?:that|this|it))(?: anymore| again)?|"
        r"(?:we|you) can stop (?:bringing (?:that|this|it) up|"
        r"talking about (?:that|this|it)|mentioning (?:that|this|it))|"
        r"forget about (?:that|this|it))\b",
        re.IGNORECASE,
    )),
    ("ask", re.compile(
        r"\b(?:don'?t ask me (?:that|about that|about this|about it)(?: again)?|"
        r"do not ask me (?:that|about that|about this|about it)(?: again)?|"
        r"stop asking(?: me)?(?: about (?:that|this|it))?|"
        r"no more questions(?: about (?:that|this|it))?)\b",
        re.IGNORECASE,
    )),
    ("roast", re.compile(
        r"\b(?:don'?t roast me about (?:that|this|it)|"
        r"do not roast me about (?:that|this|it)|"
        r"stop joking about (?:that|this|it)|"
        r"stop teasing me about (?:that|this|it))\b",
        re.IGNORECASE,
    )),
    # "Change the subject" family: a request to drop the CURRENT topic. Anchored
    # on a steering verb/lead-in + the subject/topic nouns (or "something else")
    # so embedded mentions ("the new subject I'm studying", "a change of subject
    # in my thesis", "let's talk about astronomy") do NOT false-trigger.
    # detect_boundary resolves the banned topic from the fallback (the live thread)
    # and tags the result kind="subject_change" — it is a TRANSIENT steer, not a
    # durable consent boundary (see interaction._handle_conversation_boundary).
    ("mention", re.compile(
        # lead-in + verb + subject/topic
        r"\b(?:let'?s|lets|can we|could we|how about we|why don'?t we|shall we|"
        r"please can we)\s+(?:choose|pick|change|switch|move on to|talk about)\s+"
        r"(?:to\s+)?(?:a\s+|an\s+|the\s+)?(?:different|new|another)?\s*(?:subject|topic)\b|"
        # bare imperative verb + a/the different|new subject/topic
        r"\b(?:choose|pick|change|switch)\s+(?:to\s+)?(?:a\s+|an\s+|the\s+)?"
        r"(?:different|new|another)\s+(?:subject|topic)\b|"
        r"\bchange the subject\b|"
        r"\btalk about something else\b|\bcan we talk about something else\b|"
        r"\bsomething else please\b|"
        # standalone "new subject" / "different topic" only at the START
        r"^\s*(?:new|different|another)\s+(?:subject|topic)\b",
        re.IGNORECASE,
    )),
]

_CLEAR_PAT = re.compile(
    r"\b(?:you can|it's okay to|it is okay to|feel free to|you may)\s+"
    r"(?P<behavior>ask|mention|roast|tease|joke about|talk about)\s+"
    r"(?:me\s+)?(?:about\s+|on\s+)?(?P<topic>[^.?!,;]+)",
    re.IGNORECASE,
)
_GENERIC_CLEAR_PAT = re.compile(
    r"\b(?:you can|it's okay to|it is okay to|feel free to|you may)\s+"
    r"(?P<behavior>ask|mention|roast|tease|joke about|talk about)\s+"
    r"(?:me\s+)?(?:about\s+|on\s+)?(?:that|this|it)(?: again)?\b",
    re.IGNORECASE,
)

_TRAILING_JUNK = re.compile(
    r"\s+(again|anymore|any more|please|okay|ok|with me|to me)$",
    re.IGNORECASE,
)


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def add_boundary(
    person_id: int,
    behavior: str,
    topic: str,
    *,
    description: Optional[str] = None,
    source_text: str = "",
) -> Optional[int]:
    behavior = _normalize_behavior(behavior)
    topic = _normalize_topic(topic)
    if not behavior or not topic:
        return None

    desc = description or _description_for(behavior, topic)
    now = _now()
    existing = db.fetchone(
        "SELECT id FROM person_conversation_boundaries "
        "WHERE person_id = ? AND behavior = ? AND topic = ?",
        (int(person_id), behavior, topic),
    )
    if existing:
        db.execute(
            "UPDATE person_conversation_boundaries "
            "SET description = ?, source_text = ?, active = 1, updated_at = ? "
            "WHERE id = ?",
            (desc, source_text.strip(), now, int(existing["id"])),
        )
        return int(existing["id"])
    return db.execute(
        "INSERT INTO person_conversation_boundaries "
        "(person_id, behavior, topic, description, source_text, active, created_at, updated_at) "
        "VALUES (?, ?, ?, ?, ?, 1, ?, ?)",
        (int(person_id), behavior, topic, desc, source_text.strip(), now, now),
    )


def deactivate_boundary(person_id: int, behavior: str, topic: str) -> None:
    db.execute(
        "UPDATE person_conversation_boundaries SET active = 0, updated_at = ? "
        "WHERE person_id = ? AND behavior = ? AND topic = ?",
        (_now(), int(person_id), _normalize_behavior(behavior), _normalize_topic(topic)),
    )


def get_boundaries(person_id: int, active_only: bool = True) -> list[dict]:
    clause = "AND active = 1 " if active_only else ""
    rows = db.fetchall(
        "SELECT * FROM person_conversation_boundaries "
        "WHERE person_id = ? "
        f"{clause}"
        "ORDER BY updated_at DESC, created_at DESC",
        (int(person_id),),
    )
    return [dict(r) for r in rows]


def summarize_for_prompt(person_id: int) -> str:
    rows = get_boundaries(person_id, active_only=True)
    if not rows:
        return ""
    lines = [
        f"- {row['description']}"
        for row in rows[:8]
        if row.get("description")
    ]
    if not lines:
        return ""
    return (
        "Conversation boundaries/preferences for this person:\n"
        + "\n".join(lines)
        + "\nThese are consent boundaries, not jokes. Follow them even when roasting."
    )


def is_blocked(person_id: int, behavior: str, topic: str) -> bool:
    behavior = _normalize_behavior(behavior)
    topic = _normalize_topic(topic)
    for row in get_boundaries(person_id, active_only=True):
        row_behavior = row.get("behavior") or ""
        row_topic = row.get("topic") or ""
        if row_topic == "anything":
            if row_behavior == "mention":
                return True
            if row_behavior == behavior:
                return True
            if behavior == "roast" and row_behavior in {"roast", "mention"}:
                return True
            continue
        if row_topic == "questions" and row_behavior == "ask" and behavior == "ask":
            return True
        if row_behavior not in {behavior, "mention"} and behavior != "roast":
            continue
        if behavior == "roast" and row_behavior not in {"roast", "mention"}:
            continue
        if _topics_overlap(topic, row_topic):
            return True
    return False


# Scaffolding words from boundary phrasings that are NOT the topic itself — stripped so
# a stored value like "do not bring up his mother" yields the topic token {mother}.
_MUTE_PHRASE_STOP = {
    "do", "not", "dont", "don", "please", "never", "ever", "again", "stop", "keep",
    "bring", "brought", "up", "talk", "talking", "talked", "mention", "mentioning",
    "comment", "commenting", "ask", "asking", "raise", "raising", "discuss",
    "discussing", "about", "the", "and", "for", "his", "her", "their", "your", "our",
    "him", "them", "topic", "subject", "stuff", "thing", "things", "anymore", "any",
    "more", "with", "over", "this", "that",
}


def _topic_terms(text: str) -> set[str]:
    return {
        t for t in re.findall(r"[a-z0-9]+", (text or "").lower())
        if len(t) >= 3 and t not in _MUTE_PHRASE_STOP
    }


def muted_topic_terms(person_id: int) -> set[str]:
    """Topic tokens Rex has been asked NOT to bring up — from active conversation
    boundaries (a mention/bring-up behavior) AND boundary/avoids preferences. Used to
    suppress matching facts from PROACTIVE prompt injection so a 'don't bring up my
    mother' boundary actually mutes the mother fact. The fact stays in the DB and
    direct recall still reads it. 'Don't ASK / don't ROAST' boundaries are excluded —
    Rex may still know the fact, he just won't ask about or tease it."""
    terms: set[str] = set()
    try:
        for row in get_boundaries(person_id, active_only=True):
            if _normalize_behavior(row.get("behavior") or "") != "mention":
                continue
            topic = _normalize_topic(row.get("topic") or "")
            if topic in {"anything", "questions", "how are you", _DEFAULT_TOPIC}:
                continue
            terms |= _topic_terms(topic)
    except Exception as exc:
        _log.debug("muted_topic_terms boundary scan failed: %s", exc)
    try:
        from memory import preferences as preferences_db
        for pref in preferences_db.find_preference(person_id):
            if (pref.get("preference_type") or "") not in {"boundary", "avoids"}:
                continue
            key = (pref.get("key") or "").replace("_topic", " ").replace("_", " ")
            terms |= _topic_terms(key)
            terms |= _topic_terms(pref.get("value") or "")
    except Exception as exc:
        _log.debug("muted_topic_terms preference scan failed: %s", exc)
    return terms


def detect_boundary(
    text: str,
    *,
    fallback_topic: Optional[str] = None,
) -> Optional[dict]:
    cleaned = (text or "").strip()
    if not cleaned:
        return None

    clear = _CLEAR_PAT.search(cleaned)
    if clear:
        topic = _normalize_topic(clear.group("topic"))
        if topic in {"again", "anymore", "any more"}:
            topic = _normalize_topic(fallback_topic or _DEFAULT_TOPIC)
        return {
            "action": "clear",
            "behavior": _normalize_behavior(clear.group("behavior")),
            "topic": topic,
            "source_text": cleaned,
        }
    clear_generic = _GENERIC_CLEAR_PAT.search(cleaned)
    if clear_generic:
        return {
            "action": "clear",
            "behavior": _normalize_behavior(clear_generic.group("behavior")),
            "topic": _normalize_topic(fallback_topic or _DEFAULT_TOPIC),
            "source_text": cleaned,
        }

    for behavior, pattern in _BOUNDARY_PATTERNS:
        match = pattern.search(cleaned)
        if not match:
            continue
        topic = _normalize_topic(match.groupdict().get("topic") or fallback_topic or _DEFAULT_TOPIC)
        if topic in {"again", "anymore", "any more"}:
            topic = _normalize_topic(fallback_topic or _DEFAULT_TOPIC)
        elif topic == _DEFAULT_TOPIC and fallback_topic:
            # A pronoun object normalized to the placeholder — the live thread
            # knows the real subject better than "current topic" does.
            topic = _normalize_topic(fallback_topic)
        return {
            "action": "add",
            "behavior": behavior,
            "topic": topic,
            "kind": "boundary",
            "description": _description_for(behavior, topic),
            "source_text": cleaned,
        }
    for behavior, pattern in _GENERIC_BOUNDARY_PATTERNS:
        if not pattern.search(cleaned):
            continue
        topic = _normalize_topic(fallback_topic or ("anything" if behavior == "roast" else _DEFAULT_TOPIC))
        # The change-the-subject family (the LAST generic pattern) is a TRANSIENT
        # steer: the caller pivots the conversation instead of storing a durable
        # consent boundary about whatever the thread label happened to be.
        subject_change = pattern is _GENERIC_BOUNDARY_PATTERNS[-1][1]
        return {
            "action": "add",
            "behavior": behavior,
            "topic": topic,
            "kind": "subject_change" if subject_change else "boundary",
            "description": _description_for(behavior, topic),
            "source_text": cleaned,
        }
    return None


def _record_boundary_episode(person_id: int, behavior: str, topic: str, action: str) -> None:
    """Log a boundary set/cleared to Rex's episodic memory ("Bret asked me not to
    bring up X"). Lazy imports keep memory/ DAG-clean; gated + failure-safe so a diary
    hiccup never blocks a consent change."""
    try:
        from memory import episodes
        name = None
        try:
            from memory import people as _people
            row = _people.get_person(person_id)
            if row is not None:
                name = row.get("name")
        except Exception:
            name = None
        episodes.record_boundary(
            person_id if isinstance(person_id, int) else None,
            behavior, topic, action, person_name=name,
        )
    except Exception as exc:
        _log.debug("[boundaries] episodic boundary capture failed: %s", exc)


def apply_detected_boundary(person_id: int, detected: dict) -> Optional[dict]:
    if not detected:
        return None
    action = detected.get("action")
    behavior = detected.get("behavior") or "mention"
    topic = detected.get("topic") or _DEFAULT_TOPIC
    if action == "clear":
        deactivate_boundary(person_id, behavior, topic)
        _log.info(
            "[boundaries] cleared boundary person_id=%s behavior=%s topic=%s",
            person_id, behavior, topic,
        )
        _record_boundary_episode(person_id, behavior, topic, "clear")
        return {"action": "clear", "behavior": behavior, "topic": topic}
    if action == "add":
        row_id = add_boundary(
            person_id,
            behavior,
            topic,
            description=detected.get("description"),
            source_text=detected.get("source_text", ""),
        )
        _log.info(
            "[boundaries] saved boundary id=%s person_id=%s behavior=%s topic=%s",
            row_id, person_id, behavior, topic,
        )
        # "Stop bringing up X" must also stop the proactive check-ins about X —
        # otherwise the celebration/emotional greeting keeps leading with a
        # remembered event (e.g. "your back pain is improving") the person just
        # asked Rex to drop. Only the "don't ask / don't mention" behaviors (not
        # "roast"); token-overlap means a vague topic mutes nothing.
        if behavior in {"ask", "mention"} and _boundary_mutes_events():
            try:
                from memory import emotional_events as _emo
                muted = _emo.mute_matching_positive_events(
                    person_id, topic, reason=f"boundary: {behavior} {topic}"
                )
                if muted:
                    _log.info(
                        "[boundaries] muted %d check-in event(s) for topic %r",
                        len(muted), topic,
                    )
            except Exception as exc:
                _log.debug("[boundaries] event-mute on boundary failed: %s", exc)
        # Any consent boundary also retires matching banked callback-humor
        # premises — "stop asking about my job" makes a job joke tone-deaf
        # too, so all three behaviors retire. Retire, not delete: the memory
        # itself stays; not config-gated because consent isn't a tunable.
        # (Boundaries from PRIOR sessions are enforced read-side: the engine
        # re-checks is_blocked per premise at fire time.)
        try:
            from memory import callbacks as _callbacks
            _callbacks.retire_matching_topic(
                person_id, topic, reason=f"boundary: {behavior} {topic}"
            )
        except Exception as exc:
            _log.debug("[boundaries] callback retire on boundary failed: %s", exc)
        _record_boundary_episode(person_id, behavior, topic, "add")
        return {"action": "add", "id": row_id, "behavior": behavior, "topic": topic}
    return None


def _boundary_mutes_events() -> bool:
    try:
        import config
        return bool(getattr(config, "BOUNDARY_MUTES_MATCHING_EVENTS", True))
    except Exception:
        return True


def reconcile_event_mutes(person_id: int) -> int:
    """Apply this person's EXISTING active 'don't bring up X' boundaries to their
    remembered events — mute check-ins for events matching each boundary topic.

    Set-time muting (apply_detected_boundary) only covers boundaries set AFTER that
    code existed; a boundary stored in a prior session would otherwise never mute
    the event. Idempotent (already-muted events are skipped by the muter), cheap, and
    safe to call on first-sight before picking a celebration to lead with."""
    if not _boundary_mutes_events():
        return 0
    try:
        from memory import emotional_events as _emo
    except Exception:
        return 0
    total = 0
    for boundary in get_boundaries(person_id, active_only=True):
        if (boundary.get("behavior") or "") not in {"ask", "mention"}:
            continue
        topic = (boundary.get("topic") or "").strip()
        if not topic:
            continue
        try:
            total += len(_emo.mute_matching_positive_events(
                person_id, topic,
                reason=f"boundary: {boundary.get('behavior')} {topic}",
            ))
        except Exception as exc:
            _log.debug("[boundaries] reconcile mute failed topic=%r: %s", topic, exc)
    return total


def _normalize_behavior(value: str) -> str:
    v = (value or "").strip().lower()
    if v in {"tease", "joke about", "make fun of"}:
        return "roast"
    if v in {"talk about", "bring up", "comment on"}:
        return "mention"
    if v in {"question"}:
        return "ask"
    return v or "mention"


def _normalize_topic(value: str) -> str:
    topic = (value or _DEFAULT_TOPIC).strip().lower()
    topic = re.sub(r"^(me\s+)?(about|for|over|on)\s+", "", topic)
    topic = re.sub(r"^(my|the|that|it|this)\s+", "", topic)
    topic = _TRAILING_JUNK.sub("", topic).strip()
    topic = re.sub(r"\s+", " ", topic)
    return _TOPIC_ALIASES.get(topic, topic or _DEFAULT_TOPIC)


def _description_for(behavior: str, topic: str) -> str:
    behavior = _normalize_behavior(behavior)
    topic = _normalize_topic(topic)
    if topic == "anything":
        if behavior == "roast":
            return "Do not roast or tease them."
        if behavior == "ask":
            return "Do not proactively ask them questions."
        return "Do not proactively mention or continue that topic."
    if topic == "questions":
        return "Do not proactively ask them personal questions."
    if topic == "how are you":
        return "Do not proactively ask how they are doing."
    if behavior == "roast":
        return f"Do not roast or tease them about {topic}."
    if behavior == "ask":
        return f"Do not proactively ask them about {topic}."
    return f"Do not proactively mention or comment on {topic}."


def _topics_overlap(a: str, b: str) -> bool:
    a = _normalize_topic(a)
    b = _normalize_topic(b)
    if a == b:
        return True
    clusters = [
        {"appearance", "body", "clothing", "hair", "shirt", "clothes"},
        {"face", "appearance", "identity"},
        {"voice", "identity"},
        {"work", "job", "boss", "office"},
        {"how are you", "feelings", "mood", "check in"},
        {"questions", "personal questions"},
    ]
    return any(a in cluster and b in cluster for cluster in clusters)
