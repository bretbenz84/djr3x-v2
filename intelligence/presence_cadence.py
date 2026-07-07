"""
intelligence/presence_cadence.py — presence-gated proactive speech cadence.

Over-talk and under-engagement are the same knob (owner direction 2026-07-06):
never a single global idle clamp, but a gap that scales with PRESENCE —
persistent and playful while someone is actually there with him, measured when
they're around but not chatting, quiet when the room empties.

Tier: how long since Rex's last PROACTIVE line before another chatter-class
line (idle banter, small talk, visual curiosity, room remarks...) may fire:

  engaged   conversation actively flowing        -> CONSCIOUSNESS_PROACTIVE_MIN_GAP_SECS (12s)
  present   someone visible, no live conversation -> PROACTIVE_GAP_PRESENT_IDLE_SECS (45s)
  empty     nobody visible                        -> PROACTIVE_GAP_EMPTY_ROOM_SECS (600s)

Event-driven purposes (greetings, wave-backs, identity asks, emotional
check-ins) are NOT clamped — reacting to a person arriving or waving is not
chatter, and delaying a greeting 45s would be worse than any over-talk.
The clamp is enforced centrally in action_governor._score_candidate, so it
covers candidates submitted through EVERY path (the historical leak: idle
banter's submit_external candidates carried none of the cooldown metadata the
governor's old gate read, so its priority-50 candidate faced no cadence check
at all and won every empty cycle).
"""

import logging

import config

_log = logging.getLogger(__name__)


def _visible_person_present() -> bool:
    """True when at least one person is visibly present (face or bound pose)."""
    try:
        from world_state import world_state
        for person in world_state.get("people") or []:
            if not isinstance(person, dict):
                continue
            if person.get("face_visible") or person.get("face_box") or person.get("pose"):
                return True
    except Exception:
        pass
    return False


def effective_min_gap_secs(profile=None) -> float:
    """Seconds Rex should hold off between unprompted chatter-class lines, given
    the current social situation. Falls back to the engaged-tier base on any
    doubt (a wrongly-short gap is livelier-than-intended; a wrongly-long gap in
    conversation reads as broken)."""
    base = float(getattr(config, "CONSCIOUSNESS_PROACTIVE_MIN_GAP_SECS", 12.0) or 0.0)
    if not bool(getattr(config, "PROACTIVE_CADENCE_CLAMP_ENABLED", True)):
        return base
    try:
        if profile is not None and getattr(profile, "conversation_active", False):
            return base
        if _visible_person_present():
            return float(getattr(config, "PROACTIVE_GAP_PRESENT_IDLE_SECS", 45.0) or base)
        # likely_still_present: face briefly lost but they're audibly mid-sentence —
        # treat as present, not as an empty room.
        if profile is not None and getattr(profile, "likely_still_present", False):
            return float(getattr(config, "PROACTIVE_GAP_PRESENT_IDLE_SECS", 45.0) or base)
        return float(getattr(config, "PROACTIVE_GAP_EMPTY_ROOM_SECS", 600.0) or base)
    except Exception as exc:
        _log.debug("presence cadence tier failed: %s", exc)
        return base
