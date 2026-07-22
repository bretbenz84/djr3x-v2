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

# Answer-capture latch: {"armed_at", "turns_left"}. One pending room answer is plenty.
_latch: Optional[dict] = None
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


def _enabled() -> bool:
    return bool(getattr(config, "PLACE_QUESTIONS_ENABLED", True))


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


def note_asked() -> None:
    """Mark the place question asked and arm the answer-capture latch."""
    global _latch, _last_asked_at
    _last_asked_at = time.monotonic()
    _latch = {
        "armed_at": time.monotonic(),
        "turns_left": int(getattr(config, "PLACE_QUESTION_ANSWER_TURNS", 3)),
    }


# ── NAME (answer / declaration capture) ─────────────────────────────────────────

def _normalize(name: str) -> str:
    n = " ".join(str(name or "").split()).strip(" .!,;:").lower()
    for art in _ARTICLES:
        if n.startswith(art):
            n = n[len(art):]
            break
    return n.strip()


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
    return n


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
              "refreshed known" if known else "learned", name, place_id)
    return dict(_last_capture)


def _enroll(name: str) -> Optional[int]:
    svc = _service()
    if svc is None:
        return None
    try:
        return svc.enroll(name)
    except Exception as exc:
        _log.debug("[place_questions] enroll failed: %s", exc)
        return None


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
