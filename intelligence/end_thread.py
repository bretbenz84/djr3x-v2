"""
intelligence/end_thread.py - session-local end-of-thread grace.

Sometimes the best conversational move is to let a thread land. This module
detects user closure cues and gives Rex a short grace period where optional
follow-ups, visual curiosity, and idle chatter back off.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
import logging
import re
import threading
import time
from typing import Optional

import config

_log = logging.getLogger(__name__)


# Strong closure cues — explicit goodbyes / "we're done here" signals only.
# Bare politeness ("thanks", "thank you", "sounds good", "got it", "cool") was
# REMOVED from here: it falsely closed conversational replies like "Well thank
# you" (a reply to a compliment, not a goodbye), arming the grace window that
# muzzled Rex's proactive re-engagement. Those soft acks now close the thread
# ONLY via _SHORT_ACK_PAT below — i.e. when the WHOLE utterance is just the ack
# AND Rex had just asked a question.
_CLOSURE_PAT = re.compile(
    r"\b(that'?s all|that is all|that'?s it|that is it|all good|i'?m good|"
    r"i am good|nothing else|no more|let'?s leave it there|leave it there|"
    r"we can stop|let'?s stop|moving on|anyway,? never ?mind|never ?mind|"
    r"don'?t want to talk about (?:it|that|this)(?: anymore| again)?|"
    r"do not want to talk about (?:it|that|this)(?: anymore| again)?|"
    r"i told you i didn'?t want to talk about (?:it|that|this)|"
    r"i told you i did not want to talk about (?:it|that|this)|"
    r"bye|goodbye|good-bye|"
    r"see you|see ya|later|talk to you later|talk later|nice speaking|"
    r"nice talking|nice chatting|"
    r"i'?m\s+(?:gonna|going\s+to)\s+(?:go|leave|head\s+out|step\s+out|take\s+off)|"
    r"i\s+am\s+going\s+to\s+(?:go|leave)|i\s+have\s+to\s+(?:go|leave)|gotta\s+go|"
    r"leaving\s+(?:the\s+room|now)|stepping\s+out|be\s+right\s+back|"
    r"i'?ll\s+be\s+(?:right\s+)?back|(?:going|heading|off)\s+to\s+bed)\b",
    re.IGNORECASE,
)
_SHORT_ACK_PAT = re.compile(
    r"^\s*(ok|okay|cool|nice|yeah|yep|alright|right|gotcha|thanks|thank you)\s*[.!]?\s*$",
    re.IGNORECASE,
)
# An affirmative reply to an invitation Rex just EXTENDED is acceptance, not a
# goodbye. Field 2026-08-27 13:36:56 — Rex: "Hey Bret, I'm thinking about you.
# Want to sit with me a minute?"; Bret: "Yeah."; the short-ack lane below called it
# "short acknowledgement after Rex prompt", armed the 35s grace, and every lean
# impulse logged "blocked: end_thread_grace" for the next 47 seconds — an accepted
# invitation to sit together came out as a crash. The ack word cannot tell the two
# apart ("yeah" to "so that was the whole story, huh?" IS closure), so the
# discriminator is the REX LINE being answered, never the ack alone.
_AFFIRMATIVE_ACK_PAT = re.compile(
    r"^\s*(?:yes|yeah|yep|yup|sure|ok|okay|alright|cool|nice|right|gotcha)\s*[.!]?\s*$",
    re.IGNORECASE,
)
# Offer-shaped Rex turns. Gated on the line actually being a QUESTION (see
# _is_invitation) so a plain statement that happens to contain "up for" or
# "how about" can't fake an invitation out of ordinary banter.
_INVITATION_PAT = re.compile(
    r"(?:\bdo you want\b|\bwould you (?:like|care|mind)\b|"
    r"\bwant (?:to|me to|a|some|you to)\b|\bwanna\b|\bcare to\b|"
    r"\bup for\b|\bfeel like\b|\bmind if i\b|"
    r"\b(?:shall|should|can|could|may) (?:i|we)\b)",
    re.IGNORECASE,
)
# The "stay with me" family reads as an invitation even without a question mark
# ("Stick around, I'm not done"), so it bypasses the question gate.
_PRESENCE_INVITE_PAT = re.compile(
    r"\b(?:sit (?:with|by|next to) me|come sit|stick around|"
    r"stay (?:a|for a|another) (?:minute|bit|while|sec|second)|"
    r"hang (?:out|with me)|keep me company|join me)\b",
    re.IGNORECASE,
)
# ...and the offers that ARE goodbyes. "Should I let you get back to it?" and
# "Let's leave it there" are offer-shaped too, and a "yeah" to those genuinely ends
# the thread — so they VETO the invitation read instead of suppressing closure.
_RELEASE_OFFER_PAT = re.compile(
    r"\b(?:let you (?:go|get back|be)|leave you (?:to it|alone)|"
    r"get out of your (?:hair|way)|call it (?:a day|a night|here|there|quits)|"
    r"wrap (?:this|it) up|sign off|shut up|be quiet|stop talking|"
    r"leave it there|moving on|go to sleep)\b",
    re.IGNORECASE,
)
# Genuine sign-offs — "I'm leaving / signing off" cues, a deliberately narrower
# subset of _CLOSURE_PAT. Only these arm the farewell-departure latch: a topic
# closure like "moving on" or "that's all" should NOT make Rex go dormant if the
# person then steps off-camera, but an actual goodbye should.
_FAREWELL_PAT = re.compile(
    r"\b(bye|goodbye|good-bye|see you|see ya|"
    r"talk to you later|talk later|catch you later|"
    r"nice (?:talking|chatting|speaking)|"
    # "leave"/"step out" verbs added 2026-07-11: "I'm gonna leave the room now"
    # matched NOTHING here, so the whole farewell-departure latch never armed and
    # Rex kept asking questions at an empty room for 2+ minutes (field-logged).
    r"i'?m\s+(?:gonna|going to)\s+(?:go|leave|head out|step out|take off|get going)|"
    r"i\s+(?:gotta|have to|need to|hafta)\s+(?:go|leave|head out|take off|get going|run)|"
    r"gotta\s+go|heading\s+out|i'?m\s+off|i'?m\s+out|i'?m\s+leaving|"
    r"leaving\s+(?:the\s+room|now)|stepping\s+out|"
    r"be\s+right\s+back|\bbrb\b|i'?ll\s+be\s+(?:right\s+)?back|"
    r"(?:going|heading|off)\s+to\s+bed|"
    r"take care)\b",
    re.IGNORECASE,
)
_THANKS_FOR_ASKING_PAT = re.compile(r"\bthanks?(?:\s+you)?\s+for\s+asking\b", re.IGNORECASE)
_QUESTION_START = re.compile(
    r"^\s*(who|what|when|where|why|how|can|could|would|will|do|does|did|"
    r"is|are|am|should)\b",
    re.IGNORECASE,
)
_SWITCH_PAT = re.compile(
    r"\b(by the way|speaking of|new subject|change the subject|let'?s talk about)\b",
    re.IGNORECASE,
)

# Proactive purposes that are about the empty ROOM or Rex HIMSELF — bored grumbles,
# empty-room riffs, ambient observations — as opposed to re-engaging the person or
# rehashing the conversation. These are the only things allowed to run once a
# farewell has closed the conversation: they keep Rex company in an empty room and
# carry him into the doze-off-to-sleep flow, without ever continuing the chat the
# person just ended. (idle_monologue is shared with the person-facing idle banter,
# but that path self-suppresses on is_grace_active() before it ever reaches here.)
_EMPTY_ROOM_PURPOSES = frozenset({
    "idle_monologue",
    "ambient_observation",
    "startup_empty_room",
})


@dataclass
class EndThreadState:
    closing_pending: bool
    reason: str
    user_text: str
    quiet_until: float
    detected_at: float


_lock = threading.Lock()
_state: Optional[EndThreadState] = None
_last_assistant_had_question: bool = False
_last_assistant_text: str = ""
# Monotonic time an affirmative reply accepted a Rex invitation. Read ONCE by the
# agenda (consume_invitation_acceptance) so a stale flag can never make a later
# turn read as an acceptance.
_invitation_accepted_at: Optional[float] = None
# Monotonic time of the last explicit verbal farewell, and the latch set once that
# farewell is followed by the person leaving the camera view. While the latch is
# live, Rex treats the conversation as fully closed — no proactive re-engagement —
# until they come back (a new turn or a presence return) or the safety cap lapses.
_farewell_at: Optional[float] = None
_conversation_closed_at: Optional[float] = None


def clear() -> None:
    global _state, _last_assistant_had_question, _farewell_at, _conversation_closed_at
    global _last_assistant_text, _invitation_accepted_at
    with _lock:
        _state = None
        _last_assistant_had_question = False
        _last_assistant_text = ""
        _invitation_accepted_at = None
        _farewell_at = None
        _conversation_closed_at = None


def note_assistant_turn(text: str) -> None:
    global _last_assistant_had_question, _last_assistant_text
    cleaned = (text or "").strip()
    if not cleaned:
        return
    with _lock:
        _last_assistant_had_question = "?" in cleaned
        _last_assistant_text = cleaned


def note_user_turn(
    text: str,
    person_id: Optional[int] = None,
    *,
    answered_question: Optional[dict] = None,
) -> Optional[dict]:
    global _invitation_accepted_at
    cleaned = (text or "").strip()
    if not cleaned:
        return None

    # An acceptance belongs to the turn that armed it. The agenda only consumes it
    # on turns that reach build_turn_plan, and plenty of turns return before that
    # (the face-reveal ask, the off-camera identify ask, repair/game/router acks) —
    # so an unread flag must not colour the NEXT turn, where a real "never mind"
    # has to be free to close the thread. _closure_reason re-arms it below when
    # this turn is genuinely an acceptance.
    with _lock:
        _invitation_accepted_at = None

    if _starts_new_thread(cleaned):
        clear()
        return None

    reason = _closure_reason(
        cleaned, answered_question=answered_question, person_id=person_id
    )
    if not reason:
        # A real new user turn means the old grace period has done its job.
        if len(re.findall(r"[A-Za-z']+", cleaned)) >= 4:
            clear()
        return None

    now = time.monotonic()
    state = EndThreadState(
        closing_pending=True,
        reason=reason,
        user_text=cleaned,
        quiet_until=now + _grace_secs(),
        detected_at=now,
    )
    is_farewell = bool(_FAREWELL_PAT.search(cleaned))
    global _state, _farewell_at
    with _lock:
        _state = state
        if is_farewell:
            _farewell_at = now
    return asdict(state)


def mark_closure_spoken() -> None:
    global _state
    with _lock:
        if _state is not None:
            _state.closing_pending = False
            _state.quiet_until = max(_state.quiet_until, time.monotonic() + _grace_secs())


def snapshot() -> Optional[dict]:
    with _lock:
        if _state is None:
            return None
        return asdict(_state)


def pending_closure() -> Optional[dict]:
    with _lock:
        if _state is None or not _state.closing_pending:
            return None
        if time.monotonic() > _state.quiet_until:
            return None
        return asdict(_state)


def recent_farewell(within_secs: Optional[float] = None) -> bool:
    """True when the user gave an explicit verbal goodbye recently enough that a
    camera departure now should be read as 'they said bye and left.'"""
    window = _farewell_window_secs() if within_secs is None else float(within_secs)
    with _lock:
        if _farewell_at is None:
            return False
        return (time.monotonic() - _farewell_at) <= window


def note_farewell_departure() -> bool:
    """Latch the conversation closed: the person left the camera view shortly after
    an explicit goodbye. Keeps Rex from re-engaging an empty room until they come
    back. Returns True if it latched (i.e. a recent farewell was on record)."""
    now = time.monotonic()
    global _conversation_closed_at
    with _lock:
        if _farewell_at is None or (now - _farewell_at) > _farewell_window_secs():
            return False
        _conversation_closed_at = now
        return True


def note_presence_return() -> None:
    """A departed person came back into view — a clean slate. The prior thread is
    over and Rex re-greets fresh (the return reaction handles that), so drop both
    the farewell dormancy AND any lingering end-of-thread grace from the goodbye
    so normal proactive life resumes once he's said hello."""
    clear()


def is_conversation_closed() -> bool:
    """True while a farewell-then-departure has Rex dormant. Self-expires after the
    safety cap so a missed return can never wedge him permanently silent."""
    global _conversation_closed_at
    with _lock:
        if _conversation_closed_at is None:
            return False
        if (time.monotonic() - _conversation_closed_at) > _farewell_closed_max_secs():
            _conversation_closed_at = None
            return False
        return True


def is_grace_active() -> bool:
    # A closed conversation (explicit goodbye + left view) is a hard, longer-lived
    # form of grace: every inline proactive path already backs off on this flag, so
    # folding it in here muzzles idle banter / monologue / re-engagement for free.
    if is_conversation_closed():
        return True
    with _lock:
        return _state is not None and time.monotonic() < _state.quiet_until


def can_proactive_purpose(purpose: str) -> bool:
    # Conversation closed (they said bye and left): no re-engaging the person or
    # rehashing the chat — but Rex may still keep himself company with the
    # empty-room / bored commentary that eventually dozes him off to sleep. Those
    # purposes are about the room or Rex himself, never the person who left, so
    # they survive the latch; everything person- or conversation-facing does not.
    if is_conversation_closed():
        return purpose in _EMPTY_ROOM_PURPOSES
    if not is_grace_active():
        return True
    return purpose in {"emotional_checkin", "identity_prompt", "relationship_inquiry"}


def build_directive() -> str:
    state = pending_closure()
    if not state:
        return ""
    return (
        "End-of-thread grace:\n"
        f"- The user appears to be closing this thread ({state['reason']}).\n"
        "- Primary purpose: give one short landing acknowledgement. Do not ask "
        "a new question, do not pivot topics, and do not add visual curiosity. "
        "Let the silence be acceptable."
    )


def consume_invitation_acceptance() -> bool:
    """True exactly once, for the turn in which an affirmative reply accepted an
    invitation Rex extended. One-shot AND time-boxed: the agenda reads it to give
    THIS turn the companionable purpose, and neither a missed read nor a late one
    can make a later turn behave as if it were the acceptance."""
    global _invitation_accepted_at
    with _lock:
        at = _invitation_accepted_at
        _invitation_accepted_at = None
    if at is None:
        return False
    return (time.monotonic() - at) <= _companionable_secs()


def _grace_secs() -> float:
    return max(5.0, float(getattr(config, "END_OF_THREAD_GRACE_SECS", 35.0)))


def _companionable_secs() -> float:
    return max(1.0, float(getattr(config, "COMPANIONABLE_ACCEPT_WINDOW_SECS", 20.0)))


def _farewell_window_secs() -> float:
    return max(5.0, float(getattr(config, "FAREWELL_DEPART_WINDOW_SECS", 120.0)))


def _farewell_closed_max_secs() -> float:
    return max(30.0, float(getattr(config, "FAREWELL_CLOSED_MAX_SECS", 600.0)))


def _starts_new_thread(text: str) -> bool:
    return "?" in text or bool(_QUESTION_START.search(text)) or bool(_SWITCH_PAT.search(text))


def _closure_reason(
    text: str,
    *,
    answered_question: Optional[dict],
    person_id: Optional[int] = None,
) -> str:
    if _THANKS_FOR_ASKING_PAT.search(text):
        return ""
    if _CLOSURE_PAT.search(text):
        return "explicit closure cue"
    if _SHORT_ACK_PAT.match(text):
        # The invitation check runs INSIDE the short-ack lane and after the explicit
        # closure cues, so it can only ever soften the weakest closure signal — an
        # actual goodbye still closes even when Rex's last line was an invitation.
        # Wrapped: a bad frame or a regex surprise must never take down the turn.
        try:
            if _AFFIRMATIVE_ACK_PAT.match(text):
                rex_text = _rex_turn_being_answered(answered_question, person_id)
                if _is_invitation(rex_text):
                    _note_invitation_accepted(text, rex_text)
                    return ""
        except Exception as exc:
            _log.debug("end-of-thread invitation check failed: %s", exc)
        with _lock:
            last_was_question = _last_assistant_had_question
        if answered_question or last_was_question:
            return "short acknowledgement after Rex prompt"
    return ""


def _is_invitation(rex_text: str) -> bool:
    """True when this Rex line asked them to stay / do something WITH him, rather
    than winding the thread down. An offer that is itself a goodbye ("should I let
    you get back to it?") is vetoed first, because a yes to THAT really does end
    the conversation."""
    cleaned = (rex_text or "").strip()
    if not cleaned:
        return False
    if _RELEASE_OFFER_PAT.search(cleaned):
        return False
    if _PRESENCE_INVITE_PAT.search(cleaned):
        return True
    return "?" in cleaned and bool(_INVITATION_PAT.search(cleaned))


def _rex_turn_being_answered(
    answered_question: Optional[dict],
    person_id: Optional[int],
) -> str:
    """The Rex line this ack is replying to, from whichever layer actually has it.

    answered_question carries it verbatim on the live reply path (interaction
    synthesizes it from the dialogue-act frame). The frame itself is the fallback
    that matters: the PROACTIVE lanes speak through consciousness.note_rex_utterance,
    which registers a frame but never calls note_assistant_turn — and that is exactly
    the path the "want to sit with me a minute?" check-in took on 2026-08-27.
    _last_assistant_text is the last resort for plain reply-path turns."""
    if isinstance(answered_question, dict):
        text = str(answered_question.get("question_text") or "").strip()
        if text:
            return text
    try:
        from intelligence import dialogue_act
        frame = dialogue_act.active_frame(person_id=person_id)
        if frame is not None:
            text = (getattr(frame, "text", "") or "").strip()
            if text:
                return text
    except Exception:
        pass
    with _lock:
        return _last_assistant_text


def _note_invitation_accepted(ack_text: str, rex_text: str) -> None:
    global _invitation_accepted_at
    with _lock:
        _invitation_accepted_at = time.monotonic()
    _log.info(
        "[end_thread] invitation accepted — closure suppressed ack=%r rex=%r",
        ack_text,
        rex_text[:80],
    )

