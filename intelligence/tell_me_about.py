"""
intelligence/tell_me_about.py — "let me tell you about someone" pre-briefing.

Detects the intent to brief Rex on a person who is NOT present ("I'd like to
tell you about my coworker Daniel", "we've got some tea on Jeff") so the
person DB can be pre-populated before the subject ever shows up. Deliberately
separate from intelligence/introductions.py, which handles people who ARE
here: an introduction enrolls a face/voice, a briefing only builds a dossier.

This module owns the parsing, the line banks Rex speaks from, and the
gossip/fact + kindness classification of each volunteered detail. The
multi-turn flow state itself lives in intelligence/interaction.py
(_pending_tell_about), following the same pattern as the introduction flow.
"""

from __future__ import annotations

import json
import logging
import random
import re
import time
from dataclasses import dataclass
from typing import Optional

import config
from intelligence import introductions
from memory.name_validation import normalize_person_name

_log = logging.getLogger(__name__)

_client = None


def _openai_client():
    global _client
    if _client is None:
        import apikeys
        from openai import OpenAI
        _client = OpenAI(api_key=apikeys.OPENAI_API_KEY)
    return _client


# Person relationships only — pets are stored as facts on their owner, not as
# people rows, so "tell you about my dog" stays a normal conversation.
_PERSON_REL_WORDS = "|".join(
    w for w in introductions._REL_WORDS.split("|")
    if w not in ("dog", "cat", "pet")
)

_NAME_TOKEN = r"[A-Za-z][A-Za-z'\-]*"
_NAME_PHRASE = rf"{_NAME_TOKEN}(?:\s+{_NAME_TOKEN}){{0,2}}"

# First-person intent prefix + "tell you about <subject>". The prefix matters:
# "did Jennifer tell you about Daniel?" must NOT open a briefing flow.
_TELL_INTENT_PAT = re.compile(
    r"\b(?:"
    r"(?:i|we)\s*(?:'?d)?\s*(?:would\s+)?(?:like|love|want|wanted)\s+to|"
    r"(?:i|we)\s+(?:need|have|got|gotta|ought)\s+to|"
    r"(?:i|we)\s*'?ve\s+got\s+to|"
    r"(?:i|we)\s+wanna|"
    r"let\s+(?:me|us)|lemme|"
    r"(?:can|could|may)\s+(?:i|we)|"
    r"(?:i|we)\s*'?(?:m|re|am|are)\s+(?:going\s+to|gonna)|"
    r"(?:i|we)\s+(?:should|will)|(?:i|we)\s*'?ll"
    r")\s+tell\s+you\s+(?:guys\s+|all\s+|both\s+)?"
    r"(?:a\s+(?:little|bit)(?:\s+bit)?\s+|something\s+|a\s+thing\s+or\s+two\s+|more\s+|"
    r"(?:some|a\s+few|the)\s+(?:boring\s+|juicy\s+)?(?:facts|things|stuff|stories|tea|gossip|dirt)\s+)?"
    r"about\s+(?P<subject>.+)$",
    re.IGNORECASE,
)

_FILL_IN_PAT = re.compile(
    r"\b(?:fill\s+you\s+in|give\s+you\s+the\s+(?:scoop|tea|dirt|lowdown|rundown|full\s+story)|"
    r"brief\s+you)\s+(?:on|about)\s+(?P<subject>.+)$",
    re.IGNORECASE,
)

_GOSSIP_NOUN_PAT = re.compile(
    r"\b(?:i|we)\b[^.?!]{0,28}?\b(?:gossip|tea|dirt|intel)\s+(?:about|on)\s+(?P<subject>.+)$",
    re.IGNORECASE,
)

_HEAR_PAT = re.compile(
    r"\b(?:do\s+you\s+)?(?:want|wanna|like)\s+(?:to\s+)?hear\s+"
    r"(?:(?:some|the|a\s+little)\s+)?"
    r"(?:(?:gossip|tea|dirt|scoop|the\s+scoop)\s+(?:about|on)|about)\s+(?P<subject>.+)$",
    re.IGNORECASE,
)

_SHOULD_KNOW_PAT = re.compile(
    r"\byou\s+should\s+know\s+about\s+(?P<subject>.+)$",
    re.IGNORECASE,
)

# A third-person lead-in right before a cue means someone ELSE is/was telling
# ("did he fill you in on..."). Checked against the text preceding the match.
_THIRD_PARTY_LEAD_RE = re.compile(r"\b(?:he|she|they|did|didn'?t|who|does)\s*$", re.IGNORECASE)

_GOSSIP_HINT_RE = re.compile(r"\b(?:gossip|tea|dirt|juicy|scoop|intel|drama|rumou?rs?)\b", re.IGNORECASE)
_FACTS_HINT_RE = re.compile(r"\b(?:boring\s+)?facts?\b|\bbasics\b|\bbackground\b", re.IGNORECASE)

# Subjects that look like topics, not people. Backstop for the bare-name path.
_NON_PERSON_SUBJECTS = {
    "day", "week", "weekend", "night", "morning", "evening", "trip", "vacation",
    "holiday", "party", "wedding", "job", "work", "school", "college", "project",
    "problem", "situation", "idea", "plan", "plans", "house", "car", "dog", "cat",
    "pet", "weather", "game", "movie", "show", "book", "town", "city",
    "neighborhood", "life", "story", "dream", "diet", "band", "team",
}
_PRONOUN_SUBJECTS = {
    "him", "her", "them", "it", "me", "myself", "us", "ourselves", "you",
    "yourself", "everyone", "everybody",
}

_SUBJECT_REL_PAT = re.compile(
    rf"^(?:my|our)\s+(?P<rel>{_PERSON_REL_WORDS})"
    rf"(?:\s*(?:,|named|called)?\s+(?P<name>{_NAME_PHRASE}))?[\s.,!?]*$",
    re.IGNORECASE,
)
_SUBJECT_FRIEND_OF_PAT = re.compile(
    rf"^a\s+(?P<rel>friend|coworker|co-worker|colleague|neighbor|neighbour|buddy)\s+of\s+(?:mine|ours)"
    rf"(?:\s+(?:named|called)\s+(?P<name>{_NAME_PHRASE}))?[\s.,!?]*$",
    re.IGNORECASE,
)
_SUBJECT_SOMEONE_PAT = re.compile(
    r"^(?:someone|somebody)(?:\s+(?:i|we)\s+know)?[\s.,!?]*$",
    re.IGNORECASE,
)
_SUBJECT_THIS_PERSON_PAT = re.compile(
    rf"^this\s+(?:guy|girl|woman|man|person|kid|lady|dude)"
    rf"(?:\s+(?:named|called)\s+(?P<name>{_NAME_PHRASE}))?[\s.,!?]*$",
    re.IGNORECASE,
)
_SUBJECT_BARE_NAME_PAT = re.compile(
    rf"^(?P<name>{_NAME_PHRASE})[\s.,!?]*$",
)

_DECLINE_PAT = re.compile(
    r"\b(never\s*mind|don'?t\s+worry|forget\s+it|forget\s+about\s+it|"
    r"actually\s+no|skip\s+it|not\s+important|another\s+time)\b",
    re.IGNORECASE,
)
_DONE_PAT = re.compile(
    r"\b(that'?s\s+(it|all|everything|enough|about\s+it)|that\s+is\s+(it|all|everything)|"
    r"nothing\s+(else|more)|no\s+more|i'?m\s+done|we'?re\s+done|all\s+done|"
    r"that'?ll\s+do|that\s+covers\s+it|you'?re\s+all\s+caught\s+up)\b",
    re.IGNORECASE,
)
_BARE_NO_PAT = re.compile(r"^\s*(no|nope|nah|nada|not\s+really)[\s.,!]*$", re.IGNORECASE)
_DONT_KNOW_PAT = re.compile(
    r"^\s*(?:um+|uh+|hmm+|well)?[\s,]*"
    r"(?:i\s+(?:don'?t|do\s+not)\s+know|not\s+sure|nothing|can'?t\s+think\s+of\s+anything|hmm+)"
    r"[\s.,!?]*$",
    re.IGNORECASE,
)
_GENDER_WORD_RE = re.compile(
    r"\b(man|male|guy|dude|woman|female|gal|lady|boy|girl|nonbinary|non-binary)\b",
    re.IGNORECASE,
)
_GENDER_NORMALIZE = {
    "man": "man", "male": "man", "guy": "man", "dude": "man", "boy": "boy",
    "woman": "woman", "female": "woman", "gal": "woman", "lady": "woman",
    "girl": "girl", "nonbinary": "nonbinary", "non-binary": "nonbinary",
}


@dataclass
class TellAboutParse:
    is_tell_about: bool
    name: Optional[str] = None
    relationship: Optional[str] = None
    gossip_hint: bool = False
    facts_hint: bool = False
    needs_name: bool = False
    confidence: float = 0.0
    reason: str = ""


def detect(text: str) -> TellAboutParse:
    """Detect 'I want to tell you about <person who is not here>'."""
    cleaned = (text or "").strip()
    if not cleaned:
        return TellAboutParse(False, reason="empty")
    if _DECLINE_PAT.search(cleaned):
        return TellAboutParse(False, reason="decline")
    # A live introduction ("I'd like you to meet...") belongs to introductions.
    if introductions._INTRO_PAT.search(cleaned):
        return TellAboutParse(False, reason="introduction cue")

    for pat in (_TELL_INTENT_PAT, _FILL_IN_PAT, _GOSSIP_NOUN_PAT, _HEAR_PAT, _SHOULD_KNOW_PAT):
        m = pat.search(cleaned)
        if not m:
            continue
        if _THIRD_PARTY_LEAD_RE.search(cleaned[: m.start()]):
            return TellAboutParse(False, reason="third-party lead-in")
        subject = (m.group("subject") or "").strip()
        name, relationship, ok = _parse_subject(subject)
        if not ok:
            return TellAboutParse(False, reason=f"non-person subject: {subject!r}")
        return TellAboutParse(
            True,
            name=name,
            relationship=relationship,
            gossip_hint=bool(_GOSSIP_HINT_RE.search(cleaned)),
            facts_hint=bool(_FACTS_HINT_RE.search(cleaned)),
            needs_name=not bool(name),
            confidence=0.85 if name else 0.75,
            reason="tell-about cue with person subject",
        )

    return TellAboutParse(False, reason="no cue")


def _parse_subject(subject: str) -> tuple[Optional[str], Optional[str], bool]:
    """Return (name, relationship, is_person_subject) for the text after the cue."""
    cleaned = (subject or "").strip()
    if not cleaned:
        return None, None, False
    first_word = re.split(r"[\s.,!?]+", cleaned.lower())[0]
    if first_word in _PRONOUN_SUBJECTS:
        return None, None, False

    m = _SUBJECT_REL_PAT.match(cleaned)
    if m:
        rel = introductions._normalize_relationship(m.group("rel"))
        name = normalize_person_name(m.group("name") or "", allow_single=True)
        return name, rel, True

    m = _SUBJECT_FRIEND_OF_PAT.match(cleaned)
    if m:
        rel = introductions._normalize_relationship(m.group("rel"))
        name = normalize_person_name(m.group("name") or "", allow_single=True)
        return name, rel, True

    if _SUBJECT_SOMEONE_PAT.match(cleaned):
        return None, None, True

    m = _SUBJECT_THIS_PERSON_PAT.match(cleaned)
    if m:
        name = normalize_person_name(m.group("name") or "", allow_single=True)
        return name, None, True

    m = _SUBJECT_BARE_NAME_PAT.match(cleaned)
    if m:
        raw = m.group("name")
        tokens = raw.split()
        if any(t.lower() in _NON_PERSON_SUBJECTS for t in tokens):
            return None, None, False
        if tokens[0].lower() in {"the", "a", "an", "this", "that", "these", "those"}:
            return None, None, False
        # Transcripts capitalize proper nouns; require at least one
        # capitalized token so "tell you about everything" stays conversation.
        if not any(t[:1].isupper() for t in tokens):
            return None, None, False
        name = normalize_person_name(raw, allow_single=True)
        if not name:
            return None, None, False
        return name, None, True

    return None, None, False


# ─────────────────────────────────────────────────────────────────────────────
# Reply parsing while the flow is open
# ─────────────────────────────────────────────────────────────────────────────

def parse_tone_reply(text: str) -> Optional[str]:
    """Map the 'gossip or boring facts?' answer to 'gossip' | 'fact' | None."""
    cleaned = (text or "").strip()
    if not cleaned:
        return None
    gossip_m = _GOSSIP_HINT_RE.search(cleaned)
    facts_m = _FACTS_HINT_RE.search(cleaned)
    if gossip_m and facts_m:
        return "gossip" if gossip_m.start() <= facts_m.start() else "fact"
    if gossip_m:
        return "gossip"
    if facts_m:
        return "fact"
    return None


def is_decline(text: str) -> bool:
    return bool(_DECLINE_PAT.search(text or ""))


def is_done(text: str, *, allow_bare_no: bool = False) -> bool:
    cleaned = (text or "").strip()
    if not cleaned:
        return False
    if _DONE_PAT.search(cleaned):
        return True
    if allow_bare_no and _BARE_NO_PAT.match(cleaned):
        return True
    return False


def is_blank_offer(text: str) -> bool:
    """True when the teller stalls ('um, I don't know') instead of sharing."""
    return bool(_DONT_KNOW_PAT.match((text or "").strip()))


def parse_gender(text: str) -> Optional[str]:
    m = _GENDER_WORD_RE.search(text or "")
    if not m:
        return None
    return _GENDER_NORMALIZE.get(m.group(1).lower())


def flow_fresh(state: Optional[dict], *, now: Optional[float] = None) -> bool:
    if not state:
        return False
    now = time.monotonic() if now is None else now
    ttl = float(getattr(config, "TELL_ABOUT_STEP_TTL_SECS", 240.0))
    return (now - float(state.get("asked_at") or state.get("created_at") or 0.0)) <= ttl


# ─────────────────────────────────────────────────────────────────────────────
# Line banks (venue-neutral; {name} slots in for TTS)
# ─────────────────────────────────────────────────────────────────────────────

_OPENERS_WITH_NAME = [
    "Ooh, {name} intel. I love juicy details.",
    "Sweet, let's hear the T on {name}.",
    "A {name} dossier, hand-delivered to my memory banks? Honored.",
    "Recording. {name} goes in the permanent files.",
]
_TONE_QUESTION = "First, the important part: is this juicy gossip or boring facts?"

_OPENERS_NO_NAME = [
    "Sweet, let's hear the T. First things first — what's their name?",
    "Ooh, I like juicy details. Who are we talking about? Name, please.",
    "Pre-loading a human file. What's their name?",
    "A mystery organic. I file by name — what do they go by?",
]

_INVITES_GOSSIP = [
    "Okay. Spill the beans on {name}.",
    "Let's hear the tea on {name}. I promise to act surprised later.",
    "What's the scoop on {name}?",
    "Dirt on {name}. Go. I'm all audio receptors.",
]
_INVITES_FACTS = [
    "Alright, what boring tidbits do you want in my memory about {name}?",
    "Facts on {name}. Hit me.",
    "Okay, {name} basics. Go.",
    "Fine, the responsible version. What should I know about {name}?",
]

_ACKS_PLAIN = ["Noted.", "Filed.", "Logged.", "In the banks."]
_ACKS_MORE = [
    "Got it. What else?",
    "Logged. Anything else about {name}?",
    "Filed. Keep going.",
    "Noted. What else should I know about {name}?",
]
_ACKS_GOSSIP_EXTRA = ["Juicy. Keep going.", "Scandalous. Continue."]

_POINTED_QUESTIONS = [
    ("gender", "Quick basics: is {name} a man or a woman, girl or boy?"),
    ("remember", "What would you like me to remember about {name}?"),
    ("relationship", "How do you actually know {name}?"),
    ("doing", "What does {name} do — job, hobbies, suspicious talents?"),
]

_CLOSERS = [
    "Filed. When {name} shows up, I'll pretend I know nothing. It'll be flawless.",
    "Got it all. {name}'s file is officially warmer than our first meeting will be.",
    "Dossier closed. {name} won't suspect a thing.",
    "All saved. I am now unsettlingly prepared for {name}.",
]


def opener_with_name(name: str) -> str:
    return random.choice(_OPENERS_WITH_NAME).format(name=name) + " " + _TONE_QUESTION


def opener_no_name() -> str:
    return random.choice(_OPENERS_NO_NAME)


def tone_question(name: str) -> str:
    return f"{name}, got it. Now: is this juicy gossip or boring facts? I label my folders."


def invite_line(name: str, kind: str) -> str:
    bank = _INVITES_GOSSIP if kind == "gossip" else _INVITES_FACTS
    return random.choice(bank).format(name=name)


def ack_line(name: str, kind: str, details_count: int) -> str:
    if details_count % 2 == 1:
        return random.choice(_ACKS_MORE).format(name=name)
    bank = list(_ACKS_PLAIN)
    if kind == "gossip":
        bank += _ACKS_GOSSIP_EXTRA
    return random.choice(bank)


def pointed_question(index: int, name: str, *, skip_relationship: bool) -> tuple[Optional[str], Optional[str]]:
    """Return (question_key, line) for the index-th stall, or (None, None) when out."""
    questions = [
        q for q in _POINTED_QUESTIONS
        if not (skip_relationship and q[0] == "relationship")
    ]
    if index >= len(questions):
        return None, None
    key, template = questions[index]
    return key, template.format(name=name)


def closer_line(name: str, details_count: int) -> str:
    if details_count <= 0:
        return f"An empty file on {name}. Bold strategy. I'll improvise when they show up."
    return random.choice(_CLOSERS).format(name=name)


def cancel_line() -> str:
    return "Fine. The file stays empty. Mysterious."


# ─────────────────────────────────────────────────────────────────────────────
# Detail classification — gossip vs fact + kindness, for memory safety gating
# ─────────────────────────────────────────────────────────────────────────────

_CLASSIFY_PROMPT = (
    "Someone (the teller) just told a social robot one statement about a THIRD "
    'person named "{name}" who is not present. Label the statement.\n'
    "Return ONLY a JSON object:\n"
    '{{"kind": "gossip" or "fact", "kindness": <number -1.0 to 1.0>, '
    '"category": "<one of: identity, family, relationship, work, interest, '
    'preference, event, health, story, other>", "key": "<short_snake_case_label>"}}\n'
    '- kind: "gossip" = secondhand stories, opinions, judgments, drama, rumors. '
    '"fact" = neutral biographical info (job, family, hobbies, preferences, '
    "where they live).\n"
    "- kindness: -1.0 mean/derogatory, 0 neutral, +1.0 kind/complimentary.\n"
    '- key: a short reusable label like "job", "favorite_band", "tijuana_story".\n'
    "The teller said this briefing is {default_kind}.\n"
    'Statement: "{text}"'
)

_MEAN_WORDS_RE = re.compile(
    r"\b(hate[sd]?|idiot|stupid|annoying|cheat(?:s|ed|ing)?|stole|steals|liar|lies|"
    r"terrible|awful|ugly|lazy|drunk|insufferable|gross|creepy|broke|loser|"
    r"arrested|fired|dumped|jerk|rude)\b",
    re.IGNORECASE,
)
_KIND_WORDS_RE = re.compile(
    r"\b(sweet(?:est)?|kind(?:est)?|amazing|brilliant|love[sd]?|generous|wonderful|"
    r"great|hilarious|talented|smart|caring|thoughtful|best)\b",
    re.IGNORECASE,
)
_GOSSIP_MARKER_RE = re.compile(
    r"\b(apparently|i heard|rumor|rumour|supposedly|don'?t tell|between us|"
    r"word is|allegedly|secretly)\b",
    re.IGNORECASE,
)

_VALID_CATEGORIES = {
    "identity", "family", "relationship", "work", "interest",
    "preference", "event", "health", "story", "other",
}


def classify_detail(text: str, subject_name: str, default_kind: Optional[str]) -> dict:
    """Classify one volunteered detail. Always returns a usable dict."""
    fallback = _heuristic_classify(text, default_kind)
    if not getattr(config, "TELL_ABOUT_CLASSIFY_LLM_ENABLED", True):
        return fallback
    try:
        prompt = _CLASSIFY_PROMPT.format(
            name=subject_name,
            default_kind=default_kind or "unspecified",
            text=text,
        )
        resp = _openai_client().chat.completions.create(
            model=config.LLM_MODEL,
            messages=[{"role": "user", "content": prompt}],
            temperature=0,
            max_tokens=80,
            response_format={"type": "json_object"},
        )
        data = json.loads((resp.choices[0].message.content or "").strip())
    except Exception as exc:
        _log.debug("[tell_about] detail classification failed, using heuristic: %s", exc)
        return fallback

    kind = str(data.get("kind") or "").strip().lower()
    if kind not in {"gossip", "fact"}:
        kind = fallback["kind"]
    try:
        kindness = max(-1.0, min(1.0, float(data.get("kindness", 0.0))))
    except (TypeError, ValueError):
        kindness = fallback["kindness"]
    category = str(data.get("category") or "").strip().lower()
    if category not in _VALID_CATEGORIES:
        category = "other"
    key = _sanitize_key(str(data.get("key") or ""))
    return {"kind": kind, "kindness": kindness, "category": category, "key": key or None}


def _heuristic_classify(text: str, default_kind: Optional[str]) -> dict:
    cleaned = text or ""
    kindness = 0.0
    if _MEAN_WORDS_RE.search(cleaned):
        kindness = -0.6
    elif _KIND_WORDS_RE.search(cleaned):
        kindness = 0.6
    kind = default_kind if default_kind in {"gossip", "fact"} else None
    if kind is None:
        kind = "gossip" if (_GOSSIP_MARKER_RE.search(cleaned) or kindness < 0) else "fact"
    return {"kind": kind, "kindness": kindness, "category": "other", "key": None}


def _sanitize_key(raw: str) -> str:
    key = re.sub(r"[^a-z0-9]+", "_", (raw or "").strip().lower()).strip("_")
    return key[:40]
