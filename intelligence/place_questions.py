"""
intelligence/place_questions.py — the learn-by-being-told loop for the ROOM ITSELF.

Sibling of room_questions.py (which learns about OBJECTS in the room); this one learns
the room's *identity*. Same three-beat shape, same fail-safe/no-LLM discipline:

  NAME   maybe_capture_answer(text) — passively watches each human turn for a room
         being named ("this is the living room", "we're in the kitchen") or, when Rex
         has just asked, a bare answer ("the living room"). On a hit it calls
         perception.place_service.enroll(name) — the running observe loop then captures
         views and commits automatically — and returns the capture so the caller can
         acknowledge. NEVER consumes the turn on its own; the caller decides.
  ASK    next_place_question() — a lull-speaker cue offered ONLY when Rex genuinely
         doesn't recognize where he is (world_state.current_place is None) and hasn't
         asked recently. Consumed by interaction's Lean lull path.
  LATCH  note_asked() — arms a short answer-capture window after Rex asks, so a bare
         "the living room" (which he can't otherwise tell from chatter) counts, and so
         the reply-frame shields it from the person-introduction path.

Disambiguation is deliberately conservative: an UNLATCHED declarative only enrolls when
it names a known room word (config.PLACE_ROOM_WORDS), so "this is Sarah" can never mint a
room; custom names ("the lab") are accepted only as the answer to Rex's own question.

Everything is behind config.PLACE_QUESTIONS_ENABLED and no-ops when the encoder/service
isn't running. No LLM calls; a miss just leaves the turn to normal conversation.
"""

from __future__ import annotations

import logging
import random
import re
import time
from typing import Optional

import config

_log = logging.getLogger(__name__)

# Answer-capture latch: {"armed_at", "turns_left", "asked_text"}. One pending room
# answer is plenty.
_latch: Optional[dict] = None

# Rex lines that belong to the place flow and so never disarm the latch (note_rex_line).
_PLACE_FLOW_SOURCES = frozenset({
    "place_question", "place_enrollment", "place_denial", "place_drive_rule",
})
_last_asked_at: float = 0.0
_last_capture_at: float = 0.0
_last_capture: Optional[dict] = None    # {"name", "place_id"}

# Here-declaration cues ("we are HERE, in this place"). Anchored on the speaker asserting
# the current location, NOT a passing mention ("I love the kitchen at my mom's").
# Present tense is spelled out literally: an optional-apostrophe we'?re also matches the
# PAST-tense "were" ("when we were in the kitchen…" enrolled the current view as the
# kitchen — caught in review), so the contraction forms require their apostrophe here;
# the latched answer path below still tolerates Whisper's dropped apostrophes.
_DECLARE_RE = re.compile(
    r"\b(?:this\s+is|here\s+is|here'?s|this\s+room\s+is|this\s+place\s+is|"
    r"we're\s+in|we\s+are\s+in|you're\s+in|you\s+are\s+in|call\s+this|welcome\s+to)\b",
    re.IGNORECASE,
)

# Reminiscence / hypothetical guard: talking about a room in the past ("when we were in
# the kitchen", "the den used to be…") is never a statement about where Rex is NOW.
_PAST_RE = re.compile(
    r"\b(?:when|while|remember|yesterday|used\s+to|back\s+(?:in|when|then)|"
    r"last\s+(?:night|week|month|year|time))\b",
    re.IGNORECASE,
)

# Answer-shaped openers for the LATCHED path (Rex just asked "what room is this?").
# Strips leading fillers plus one declaration prefix, INCLUDING Whisper's apostrophe-less
# forms ("its the nook", "thats the den") — room_questions hit the same transcription
# quirk. Without this, "it's the nook" minted a room literally named "it's the nook".
_ANSWER_PREFIX_RE = re.compile(
    r"^(?:(?:well|oh|uh|um|hmm|hey|so|yeah|yes|okay|ok|right|i\s+think|i'?d\s+say|"
    r"probably|maybe|looks\s+like)[,\s]+)*"
    r"(?:(?:it'?s|its|it\s+is|that'?s|thats|that\s+is|this\s+is|here'?s|heres|"
    r"we'?re\s+in|were\s+in|we\s+are\s+in|you'?re\s+in|youre\s+in|you\s+are\s+in|"
    r"call\s+it|call\s+this|i\s+call\s+it)\s+)?",
    re.IGNORECASE,
)

# Deflections / non-answers — mirror room_questions so "no idea" closes the ask.
_NON_ANSWERS = (
    "i don't know", "i dont know", "no idea", "not sure", "dunno", "who knows",
    "nothing", "don't worry", "dont worry", "never mind", "nevermind",
    "none of your", "wouldn't you like to know", "why do you", "why would you",
)

_ARTICLES = ("the ", "a ", "an ", "my ", "our ", "your ")
_FILLERS = ("what", "why", "how", "who", "when", "where", "anyway", "well", "okay",
            "ok", "yeah", "yes", "no", "nope", "hmm", "huh", "sure", "right", "hey",
            "oh", "uh", "um", "so", "i", "it", "that", "this", "we", "you", "just")

# The latched bare-answer path is the loosest capture in the module: any short phrase
# that isn't a filler becomes a room name. Field 2026-08-06 — Rex asked "which room is
# this?", never got a usable answer (the real "this is the workshop" was lost to an
# echo-hallucinated transcript), moved on to a news offer 24s later, and then filed the
# reply to THAT — "Tell me more." — as a room, enrolling 8 views of the actual workshop
# under a place named "tell me more".
#
# A room name is a NOUN PHRASE ("the lab", "my studio", "garage"). These openers make a
# phrase a REQUEST aimed at Rex, which is never what a room is called. Matched on the
# FIRST WORD only, so "playroom"/"study"/"shop" are untouched, and any phrase carrying a
# known room word bypasses the veto entirely.
_IMPERATIVE_OPENERS = (
    "tell", "say", "speak", "talk", "repeat", "explain", "describe", "elaborate",
    "continue", "go", "keep", "carry", "stop", "quit", "wait", "hold", "come", "turn",
    "move", "drive", "roll", "walk", "follow", "bring", "send", "play", "sing", "dance",
    "show", "give", "let", "make", "do", "don't", "dont", "try", "check", "find",
    "look", "listen", "watch", "open", "close", "start", "begin", "pause", "skip",
    "shut", "power", "reboot", "restart", "sleep", "wake", "help", "forget", "remember",
    "read", "write", "add", "remove", "delete", "cancel", "change", "switch", "pick",
    "choose", "answer", "ask", "call", "get", "put", "take", "leave", "hurry", "relax",
    "shush", "hush", "quiet", "more", "again", "another", "anything", "something",
    "nothing", "everything", "whatever", "please", "thanks", "thank", "sorry",
)

# Whole-phrase non-names that slip past the first-word veto.
_NOT_ROOM_PHRASES = frozenset({
    "go on", "carry on", "one more time", "the same", "same thing", "not much",
    "not really", "of course", "for sure", "all good", "no thanks", "no thank you",
    "any", "some", "none", "both", "either", "neither", "us", "them", "me", "him",
    "her", "everyone", "anyone", "nobody", "somebody",
})


def _enabled() -> bool:
    return bool(getattr(config, "PLACE_QUESTIONS_ENABLED", True))


def _transcript_trusted() -> bool:
    """Whether the transcript for this turn was one Whisper actually believed.
    Field 2026-07-25: "This is the workshop room" was decoded "Shop room." and a
    room called "shop room" was created from it."""
    try:
        from intelligence.interaction import _turn_transcript_trusted
        return bool(_turn_transcript_trusted())
    except Exception:
        return True


def _room_words() -> list:
    return [str(w).strip().lower() for w in getattr(config, "PLACE_ROOM_WORDS", []) if str(w).strip()]


_ROOM_WORD_RE_CACHE: Optional[tuple] = None


def _room_word_re() -> "re.Pattern":
    """Alternation of known room words, longest-first so 'master bedroom' beats 'bedroom'."""
    global _ROOM_WORD_RE_CACHE
    words = _room_words()
    if _ROOM_WORD_RE_CACHE is None or _ROOM_WORD_RE_CACHE[0] != tuple(words):
        alt = "|".join(re.escape(w) for w in sorted(set(words), key=len, reverse=True))
        pat = re.compile(r"\b(" + alt + r")\b", re.IGNORECASE) if alt else re.compile(r"(?!x)x")
        _ROOM_WORD_RE_CACHE = (tuple(words), pat)
    return _ROOM_WORD_RE_CACHE[1]


# Head nouns a room name ends in. Derived from PLACE_ROOM_WORDS (so user_config
# additions extend it for free) plus generic spatial heads. Used to spare compound
# names from the request veto: "play room" and "show room" are rooms, "play music"
# and "show me" are not.
_EXTRA_ROOM_HEADS = ("room", "area", "space", "spot", "nook", "corner", "wing",
                     "suite", "quarters", "annex", "shed", "barn")
_ROOM_HEAD_CACHE: Optional[tuple] = None


def _room_head_nouns() -> frozenset:
    global _ROOM_HEAD_CACHE
    words = _room_words()
    if _ROOM_HEAD_CACHE is None or _ROOM_HEAD_CACHE[0] != tuple(words):
        heads = {w.split()[-1] for w in words if w.split()}
        heads.update(_EXTRA_ROOM_HEADS)
        _ROOM_HEAD_CACHE = (tuple(words), frozenset(heads))
    return _ROOM_HEAD_CACHE[1]


# ── Availability (only act when the recognizer is actually running) ──────────────

def _service():
    try:
        from perception import place_service
        return place_service
    except Exception:
        return None


def _place_available() -> bool:
    svc = _service()
    return bool(svc and svc.get_recognizer() is not None)


def _belief_known() -> bool:
    svc = _service()
    try:
        return bool(svc and svc.current_place())
    except Exception:
        return False


def _is_enrolling() -> bool:
    svc = _service()
    try:
        return bool(svc and svc.state() == "collecting")
    except Exception:
        return False


# ── ASK ──────────────────────────────────────────────────────────────────────

def next_place_question() -> Optional[dict]:
    """A lull-speaker cue asking what room this is, or None. Fires only when place
    recognition is live, Rex has NO current belief, isn't already enrolling, wasn't just
    told, and the cooldown is clear. Returns {"text": <LLM instruction>}."""
    if not _enabled() or not _place_available():
        return None
    if _belief_known() or _is_enrolling():
        return None
    cooldown = float(getattr(config, "PLACE_QUESTION_COOLDOWN_SECS", 600.0))
    now = time.monotonic()
    if (now - _last_asked_at) < cooldown:
        return None
    if (now - _last_capture_at) < cooldown:      # don't re-ask right after being told
        return None
    templates = getattr(config, "PLACE_QUESTION_TEMPLATES", None) or [
        "Ask, briefly and in character, what room you're in — you don't recognize it."
    ]
    return {"text": random.choice(list(templates))}


def note_asked(text: str = "") -> None:
    """Mark the place question asked and arm the answer-capture latch.

    `text` is the line Rex actually spoke; it exempts that line from note_rex_line()'s
    disarm so the ask can never cancel its own latch, whatever order the caller uses.
    """
    global _latch, _last_asked_at
    _last_asked_at = time.monotonic()
    _latch = {
        "armed_at": time.monotonic(),
        "turns_left": int(getattr(config, "PLACE_QUESTION_ANSWER_TURNS", 3)),
        "asked_text": " ".join(str(text or "").split()).lower(),
    }


def note_rex_line(text: str = "", source: Optional[str] = None) -> None:
    """Disarm the answer-capture latch once Rex has moved on to something else.

    The latch only ever counted HUMAN turns, so it survived REX changing the subject.
    Field 2026-08-06: he asked "which room is this?", got nothing usable back, offered
    a news story 24s later, and then filed the reply to the NEWS ("Tell me more.") as
    the room's name. A question he has already talked past is not pending any more —
    the same rule `_awaiting_followup_event` uses when a later lull line opens a
    different thread. Lines belonging to the place flow itself keep the latch.
    """
    global _latch
    latch = _latch
    if latch is None:
        return
    if str(source or "") in _PLACE_FLOW_SOURCES:
        return
    line = " ".join(str(text or "").split()).lower()
    if line and line == latch.get("asked_text"):
        return                              # this IS the ask
    _latch = None
    _log.debug("[place_questions] answer latch dropped — Rex moved on (source=%s)", source)


# ── NAME (answer / declaration capture) ─────────────────────────────────────────

def _normalize(name: str) -> str:
    n = " ".join(str(name or "").split()).strip(" .!,;:").lower()
    for art in _ARTICLES:
        if n.startswith(art):
            n = n[len(art):]
            break
    return n.strip()


def _looks_like_a_request(phrase: str) -> bool:
    """True when a short phrase is something said TO Rex, not the name of a room.

    "Tell me more", "go on", "keep going" are replies to whatever Rex said LAST —
    they are never what a room is called. See _IMPERATIVE_OPENERS.
    """
    if phrase in _NOT_ROOM_PHRASES:
        return True
    words = phrase.split()
    return bool(words) and words[0] in _IMPERATIVE_OPENERS


def _bare_answer(text: str) -> Optional[str]:
    """A short direct reply ('the lab', 'kitchen', 'my studio') — but not chatter."""
    n = _normalize(text)
    if not n:
        return None
    words = n.split()
    if not (1 <= len(words) <= 4):
        return None
    if words[0] in _FILLERS:
        return None
    # A known room word — or simply ending in a room head noun, which is what keeps
    # "play room"/"show room" working — vouches for the phrase outright. Anything else
    # has to be shaped like a name rather than an instruction aimed at Rex.
    vouched = bool(_room_word_re().search(n)) or words[-1] in _room_head_nouns()
    if not vouched and _looks_like_a_request(n):
        return None
    return n


# "this is NOT the workshop" / "you're not in the workshop" / "that's not the kitchen".
# Deliberately narrow: an explicit negation attached to a here-statement about a room.
def _room_tail(group: str) -> str:
    return (r"(?:in\s+|at\s+)?(?:the\s+|a\s+|my\s+|our\s+)?"
            rf"(?P<{group}>[a-z][a-z' ]{{1,24}}?)")


_PLACE_DENIAL_RE = re.compile(
    # Anchored at the START on a HERE-subject, so "I do not like the workshop" (an
    # opinion) and "we're not done in the kitchen" (not about the belief) can't match.
    r"^(?:"
    r"(?:this|that|it|here)\s*(?:'s|s|is|was)?\s*(?:not|isn't|is n't|ain't)\s+"
    + _room_tail("room_a") +
    r"|(?:you|we)\s*(?:'re|re|are)?\s*(?:not|aren't|are n't)\s+(?:in|at)\s+"
    + _room_tail("room_b") +
    r")\s*[.!]*$",
    re.I,
)


def _denial_room(match) -> str:
    return (match.group("room_a") or match.group("room_b") or "") if match else ""


def maybe_capture_denial(text: str) -> Optional[dict]:
    """Consume "this is not the <room>" — drop the believed room instead of arguing.

    Field 2026-07-24: Rex answered "This is not the workshop." with "Yep, the
    workshop. I recognize it." A human standing in the room outranks a cosine score.
    Returns {"was": <dropped room>} when a belief was actually dropped, else None.
    Never enrolls anything: it only clears the belief, which re-arms the
    ask-what-room-this-is cue so the real name can be captured next.
    """
    if not _enabled() or not _place_available():
        return None
    cleaned = " ".join(str(text or "").split())
    if not cleaned or cleaned.endswith("?"):
        return None
    low = cleaned.lower()
    if " not " not in f" {low} " and "n't" not in low:
        return None
    if _PAST_RE.search(low):
        return None                       # reminiscing, not a statement about HERE
    svc = _service()
    try:
        believed = ((svc.current_place() if svc else None) or {}).get("name")
    except Exception:
        believed = None
    if not believed:
        return None
    believed_norm = _normalize(believed)
    # Require the anchored here-denial AND that it names the room Rex actually
    # believes. Precision over recall: a merely-contains-the-name test fired on "I
    # do not like the workshop", and "this isn't the garage" while believing the
    # kitchen says nothing about the kitchen.
    m = _PLACE_DENIAL_RE.search(low)
    if m is None:
        return None
    named = _normalize(_denial_room(m))
    if not named or named != believed_norm:
        return None
    try:
        if not svc.reject_belief(believed):
            return None
    except Exception as exc:
        _log.debug("[place_questions] reject_belief failed: %s", exc)
        return None
    _log.info("[place_questions] human denied the believed room %r — belief dropped",
              believed)
    return {"was": believed}


def _extract_room_name(text: str, *, latched: bool) -> Optional[str]:
    cleaned = " ".join(str(text or "").split())
    if not cleaned or cleaned.endswith("?"):
        return None
    low = cleaned.lower()
    if any(p in low for p in _NON_ANSWERS):
        return None
    if _PAST_RE.search(low):
        return None                      # reminiscence, not a statement about HERE

    room_word_match = _room_word_re().search(low)
    room_word = room_word_match.group(1).lower() if room_word_match else None
    declared = _DECLARE_RE.search(low) is not None

    # Unlatched (volunteered): require BOTH a here-declaration AND a known room word, so a
    # plain "this is Sarah" / stray chatter can never enroll a room.
    if not latched:
        return room_word if (declared and room_word) else None

    # Latched — Rex asked what room this is, so the reply is very likely the answer.
    if declared and room_word:
        return room_word
    stripped = _ANSWER_PREFIX_RE.sub("", low, count=1).strip()
    if declared or stripped != low:
        # An answer-shaped opener ("it's …", "thats …", filler) — take the room word or
        # the short custom phrase that follows it ("the nook" -> "nook").
        return room_word or _bare_answer(stripped)
    if room_word and len(low.split()) <= 5:
        return room_word                 # "the kitchen, obviously" — short and on-topic
    # A bare short phrase ("garage", "my studio") counts; an incidental room word inside
    # a longer sentence ("I told you about the kitchen once") does not.
    return _bare_answer(low)


def maybe_capture_answer(text: str) -> Optional[dict]:
    """Observe one human turn. Enroll + name the room on a hit and return the capture
    ({"name","place_id"}), else None. Never consumes the turn. Latch decays by TTL/turns."""
    global _latch, _last_capture_at, _last_capture
    if not _enabled() or not _place_available():
        return None

    latch = _latch
    if latch is not None:
        ttl = float(getattr(config, "PLACE_QUESTION_ANSWER_TTL_SECS", 120.0))
        if (time.monotonic() - latch["armed_at"]) > ttl:
            _latch = None
            latch = None

    name = _extract_room_name(text, latched=latch is not None)
    if name is None:
        if latch is not None:
            latch["turns_left"] -= 1
            if latch["turns_left"] <= 0:
                _latch = None       # nobody named it; stop watching (cooldown re-gates)
        return None

    svc = _service()
    try:
        # Re-telling the SAME room mid-capture ("this is the living room" twice) must
        # not restart the session (dropping collected views) or double-ack; a DIFFERENT
        # name is a correction and restarts on purpose.
        if (svc and svc.state() == "collecting"
                and _normalize(svc.enrolling_name() or "") == name):
            _latch = None
            return None
    except Exception:
        pass
    known = False
    try:
        known = name in {str(n).strip().lower() for n in (svc.place_names() or [])}
    except Exception:
        known = False

    _latch = None
    place_id = _enroll(name)
    if place_id is None:
        return None
    _last_capture = {"name": name, "place_id": place_id, "known": known}
    _last_capture_at = time.monotonic()
    _log.info("[place_questions] %s room %r (place_id=%s)",
              "started visual refresh for" if known else "started learning", name, place_id)
    return dict(_last_capture)


def _enroll(name: str) -> Optional[int]:
    svc = _service()
    if svc is None:
        return None
    if not _transcript_trusted():
        _log.info("[place_questions] not enrolling %r — low-confidence transcript", name)
        return None
    try:
        return svc.enroll(name)
    except Exception as exc:
        _log.debug("[place_questions] enroll failed: %s", exc)
        return None


# ── "don't drive in this room" ──────────────────────────────────────────────────
# Rex cannot pivot on carpet — under his own weight the tyres just scrub and the turn
# never completes (the firmware aborts it; motion_agency stands down after two). The
# traction detector catches that AFTER a couple of grinding attempts. This lets the
# owner say it once, up front, and have it stick to the ROOM: "this room has carpet",
# "don't move in the workshop". Persisted per place, so walking him back in re-arms it.

_DRIVE_VERB = r"(?:mov(?:e|es|ing)|driv(?:e|es|ing)|roll(?:s|ing)?|wander(?:s|ing)?|" \
              r"go(?:es|ing)?|scoot(?:s|ing)?)"

# The place tail: either a here-reference (resolved against the current belief) or a
# named room. Deliberately loose — _resolve_room does the real work.
_PLACE_TAIL = (r"(?P<place>(?:in\s+|into\s+|around\s+)?"
               r"(?:this\s+|the\s+|that\s+|my\s+|our\s+)?[a-z][a-z' ]{0,24}?)")

_FLOOR_BAD = r"(?:carpet(?:ed|ing)?|shag|a\s+rug|rugs|carpets)"
_FLOOR_OK = (r"(?:no\s+carpet|hard\s*(?:wood)?\s*floors?|hardwood|tile[ds]?|"
             r"linoleum|lino|concrete|laminate|vinyl)")

_NO_DRIVE_RES = (
    # "don't (try to) move in this room" / "don't drive in the workshop"
    re.compile(r"\b(?:don'?t|do\s+not|never|no)\s+(?:you\s+)?(?:try(?:ing)?\s+to\s+)?"
               + _DRIVE_VERB + r"\s+(?:around\s+)?" + _PLACE_TAIL + r"\s*[.!]*$", re.I),
    # "no driving in here"
    re.compile(r"\bno\s+(?:driving|moving|rolling|wandering)\s+" + _PLACE_TAIL
               + r"\s*[.!]*$", re.I),
    # "this room has carpet" / "the workshop is carpeted" / "there's carpet in here"
    re.compile(r"^" + _PLACE_TAIL + r"\s+(?:has|have|'?s|is|are)\s+(?:got\s+)?(?:a\s+)?"
               + _FLOOR_BAD + r"\s*[.!]*$", re.I),
    re.compile(r"\bthere'?s?\s+" + _FLOOR_BAD + r"\s+" + _PLACE_TAIL + r"\s*[.!]*$", re.I),
)

_CAN_DRIVE_RES = (
    # "you can move in here" / "it's fine to drive in the workshop"
    re.compile(r"\b(?:you\s+can|you\s+may|feel\s+free\s+to|it'?s\s+(?:ok|okay|fine|"
               r"alright)\s+to|ok(?:ay)?\s+to)\s+" + _DRIVE_VERB
               + r"\s+(?:around\s+)?" + _PLACE_TAIL + r"\s*[.!]*$", re.I),
    # "this room has hardwood" / "no carpet in here"
    re.compile(r"^" + _PLACE_TAIL + r"\s+(?:has|have|'?s|is|are)\s+(?:got\s+)?"
               + _FLOOR_OK + r"\s*[.!]*$", re.I),
    re.compile(r"^" + _FLOOR_OK + r"\s+" + _PLACE_TAIL + r"\s*[.!]*$", re.I),
)

# "this room", "here", "in here" — resolve against whatever he currently believes.
_HERE_WORDS = ("this room", "this place", "this one", "here", "in here", "this area",
               "the room", "this", "it")


def _resolve_room(tail: str) -> "tuple[str | None, bool]":
    """(room name, was_a_here_reference). None when the phrase points at HERE and he
    doesn't know where here is — the caller has to say so rather than guess."""
    cleaned = " ".join(str(tail or "").split()).lower().strip(" .!,")
    for lead in ("in ", "into ", "around ", "on "):
        if cleaned.startswith(lead):
            cleaned = cleaned[len(lead):].strip()
    if cleaned in _HERE_WORDS:
        return _believed_room_name(), True
    for art in _ARTICLES:
        if cleaned.startswith(art):
            cleaned = cleaned[len(art):].strip()
            break
    if cleaned in _HERE_WORDS:
        return _believed_room_name(), True
    return (cleaned or None), False


def _believed_room_name() -> Optional[str]:
    svc = _service()
    try:
        belief = svc.current_place() if svc else None
    except Exception:
        return None
    name = (belief or {}).get("name") if isinstance(belief, dict) else None
    return str(name) if name else None


def _looks_like_a_room(name: str) -> bool:
    """A room he has enrolled, or a phrase built from a known room word. Anything
    else ("forward", "closer", "the wall") is not somewhere to file a rule against."""
    low = str(name or "").strip().lower()
    if not low:
        return False
    svc = _service()
    try:
        if svc is not None and low in [str(n).lower() for n in svc.place_names()]:
            return True
    except Exception:
        pass
    return _room_word_re().search(low) is not None


def maybe_capture_drive_rule(text: str) -> Optional[dict]:
    """Consume "this room has carpet" / "don't drive in the workshop" and persist it
    against the room. Returns None when the turn isn't one of these.

    On a hit: {"name", "no_drive", "reason", "here", "applied", "known"}. `applied` is
    False when there is no room to attach it to — an unrecognized view, or a room name
    he has never enrolled. The rule is still REPORTED so the caller can stop him now
    and explain; it just can't be filed."""
    if not _enabled() or not _place_available():
        return None
    cleaned = " ".join(str(text or "").split())
    if not cleaned or cleaned.endswith("?"):
        return None
    if _PAST_RE.search(cleaned.lower()):
        return None                      # "back when the den had carpet…"

    no_drive, match = True, None
    for rx in _NO_DRIVE_RES:
        match = rx.search(cleaned)
        if match:
            break
    if match is None:
        no_drive = False
        for rx in _CAN_DRIVE_RES:
            match = rx.search(cleaned)
            if match:
                break
    if match is None:
        return None

    name, here = _resolve_room(match.groupdict().get("place") or "")
    # The tail has to be a PLACE. Without this, "don't move forward" / "don't go
    # closer" parse as a rule about a room called "forward" and consume the turn
    # that the motion router should have had.
    if not here:
        if not name or not _looks_like_a_room(name):
            return None
    reason = "carpet" if (no_drive and re.search(_FLOOR_BAD, cleaned, re.I)) else None
    svc = _service()
    known = bool(name) and name in (svc.place_names() if svc else [])
    applied = False
    if name and known and svc is not None:
        try:
            applied = bool(svc.set_no_drive(name, no_drive, reason))
        except Exception as exc:
            _log.debug("[place_questions] set_no_drive failed: %s", exc)
    # "Is this about the room he is standing in?" — a rule filed against a room he
    # isn't in must not stop him where he is.
    current = here or (bool(name) and name == _believed_room_name())
    _log.info("[place_questions] drive rule: room=%s no_drive=%s applied=%s current=%s (%r)",
              name, no_drive, applied, current, cleaned)
    return {"name": name, "no_drive": no_drive, "reason": reason, "here": here,
            "applied": applied, "known": known, "current": current}


def drive_rule_ack_line(rule: Optional[dict]) -> str:
    """Acknowledge the rule — and be honest when it could not be filed anywhere."""
    rule = rule or {}
    name = str(rule.get("name") or "").strip()
    if not rule.get("applied"):
        if rule.get("here") and not name:
            return ("Understood — wheels off. I don't know which room this is yet, "
                    "though, so tell me its name and I'll remember it for next time.")
        if name:
            return (f"Understood — wheels off. I don't know the {name} yet, so show me "
                    "around sometime and I'll remember the rule with it.")
        return "Understood — wheels off."
    where = f"the {name}" if name and not name.startswith("the ") else (name or "here")
    if rule.get("no_drive"):
        if rule.get("reason") == "carpet":
            return f"Carpet in {where}. Noted — I'll keep my wheels still in there."
        return f"No driving in {where}. Noted."
    return f"Good to know — I can roll around in {where} again."


def denial_ack_line(denial: Optional[dict]) -> str:
    """Take the correction gracefully and invite the real name — never argue."""
    was = str((denial or {}).get("was") or "that").strip()
    templates = getattr(config, "PLACE_DENIAL_ACK_TEMPLATES", None) or [
        "My mistake — scratch the {was}. Where am I, then?",
    ]
    return random.choice(list(templates)).format(was=was)


def ack_line(capture: Optional[dict]) -> str:
    """A verbatim acknowledgement for a fresh capture ('Got it — the living room.').
    A room he already knew gets the recognition variant instead of the learning one."""
    name = str((capture or {}).get("name") or "this place").strip()
    if (capture or {}).get("known"):
        templates = getattr(config, "PLACE_KNOWN_ACK_TEMPLATES", None) or [
            "The {name} — yeah, I know this one."
        ]
    else:
        templates = getattr(config, "PLACE_ENROLL_ACK_TEMPLATES", None) or [
            "Got it — the {name}. I'll remember this place."
        ]
    return random.choice(list(templates)).format(name=name)


# ── Grounding: the room-belief clause for reply prompts ─────────────────────────

def belief_clause() -> str:
    """One honest sentence about which room Rex is in, for the reply-LLM context.

    Grounded in the recognizer's belief + latest frame — NOT the conversation
    transcript (field bug 2026-07-21: with no grounding, "what room are you in?" was
    answered by parroting whichever room was last MENTIONED). Hedges when the view is
    ambiguous, admits not knowing, and says nothing at all ("") when the feature is
    off/not running — silence must not read as "he claims ignorance".
    """
    if not _enabled():
        return ""
    svc = _service()
    try:
        ctx = svc.belief_context() if svc else None
    except Exception:
        ctx = None
    if not ctx:
        return ""
    reported = (_last_capture or {}).get("name")
    age = time.monotonic() - _last_capture_at
    belief_name = (ctx.get("belief") or {}).get("name")
    if (reported and 0 <= age < float(getattr(config, "PLACE_QUESTION_COOLDOWN_SECS", 600))
            and belief_name in (None, reported)):
        return (f"Room: the user just identified this room as {reported}. "
                "Use that reported name; do not ask which room this is again. "
                "This is the user's report, not proof that visual recognition or learning succeeded.")
    enrolling = ctx.get("enrolling")
    if enrolling:
        return f"Room: you're currently memorizing what the {enrolling} looks like."
    belief = ctx.get("belief") or {}
    name = belief.get("name")
    top = ctx.get("top") or []
    # A standing "don't drive in here" belongs in the reply context too, or he'll
    # cheerfully offer to come over in a room he has been told to stay out of.
    rule = ""
    if belief.get("no_drive"):
        why = belief.get("no_drive_reason")
        rule = (" You've been told not to drive in this room"
                + (f" ({why})" if why else "")
                + " — don't offer to move or come over; say you'll stay put.")
    if name:
        if ctx.get("ambiguous") and len(top) >= 2:
            other = top[1][0] if top[0][0] == name else top[0][0]
            if other and other != name:
                return (f"Room: you believe you're in the {name}, though right now it "
                        f"looks a lot like the {other} too — hedge if asked." + rule)
        return f"Room: you're in the {name} (you recognize it).{rule}"
    if int(ctx.get("known_rooms") or 0) == 0:
        return ("Room: you don't know any rooms yet — nobody has taught you one "
                "(you learn a room when someone tells you its name).")
    return "Room: you don't recognize which room you're in right now — say so if asked."


# ── ACK helpers (parity with room_questions) ────────────────────────────────────

def recently_captured(within_secs: float = 5.0) -> bool:
    return _last_capture is not None and (time.monotonic() - _last_capture_at) <= within_secs


def last_capture() -> Optional[dict]:
    return dict(_last_capture) if _last_capture else None


def reset() -> None:
    """Test hook."""
    global _latch, _last_asked_at, _last_capture_at, _last_capture
    _latch = None
    _last_asked_at = 0.0
    _last_capture_at = 0.0
    _last_capture = None
