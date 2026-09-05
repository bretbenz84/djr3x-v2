"""
intelligence/conversation_state.py — the session's deterministic conversation facts.

Lean Brain restructuring, phase 2 (docs/lean_brain_restructuring_plan.md, "Target
ownership" #2). The reply model used to see eight transcript turns and a situation
block; everything else Rex knew about THIS conversation lived in module globals
scattered across interaction.py, or nowhere. This is the small in-memory owner for
the facts that must outrank a stale summary or a recalled assertion:

- **Corrections** the person made ("my name's not Brad", "forget that", "I didn't
  go anywhere, you turned your head", a topic boundary). Recorded by the code that
  EXECUTED the correction, so a line here is something Rex actually did.
- **Action outcomes**: what the body was asked to do and what actually happened
  (issued / completed / blocked / aborted / refused-with-reason). Recorded by
  motion_controller at issue, refusal, and done time, keyed by the firmware seq,
  so "did you turn?" is answered from the record, never from the model's guess.
- **Pending exchanges**: questions Rex asked that are still unanswered, read from
  the dialogue-act frames (which already know the target person and expected reply
  types) — rendered so another person's reply does not silently count as the
  target's answer.

Everything is session memory (cleared with topic_thread.clear()), bounded, and
rendered as a few plain lines for the prompt by `render_lines()`. Model summaries
(the conversation arc) are advisory; the lines here are stated as overriding them.
"""

from __future__ import annotations

import logging
import threading
import time
from collections import deque
from typing import Optional

import config
from intelligence.action_result import ActionResult

_log = logging.getLogger(__name__)

_lock = threading.Lock()
_corrections: "deque[dict]" = deque(maxlen=12)
_actions: "deque[ActionResult]" = deque(maxlen=12)
# Phase 2B: how sure the current turn's speaker identity is (set per turn by the
# attribution resolver, rendered for Lean, cleared with the session).
_speaker_resolution: Optional[dict] = None


def _cfg(name: str, default):
    try:
        return getattr(config, name, default)
    except Exception:
        return default


def clear() -> None:
    global _speaker_resolution
    with _lock:
        _corrections.clear()
        _actions.clear()
        _speaker_resolution = None


# ── Corrections ──────────────────────────────────────────────────────────────

def note_correction(kind: str, text: str, *, person_id: Optional[int] = None) -> None:
    """Record an explicit correction Rex acted on. `text` is a short plain-words
    statement written for the prompt ("Their name is JT, not Brad")."""
    text = " ".join(str(text or "").split())
    if not text:
        return
    entry = {
        "at": time.monotonic(),
        "kind": str(kind or "correction"),
        "text": text,
        "person_id": person_id,
    }
    with _lock:
        _corrections.append(entry)
    _log.info("[conversation_state] correction (%s): %s", entry["kind"], text)


def recent_corrections(*, max_age_secs: Optional[float] = None, limit: int = 4,
                       person_id: Optional[int] = None) -> list[dict]:
    max_age = float(max_age_secs if max_age_secs is not None
                    else _cfg("CONVERSATION_STATE_CORRECTION_TTL_SECS", 900.0))
    now = time.monotonic()
    with _lock:
        items = list(_corrections)
    out = []
    for e in reversed(items):
        if now - e["at"] > max_age:
            continue
        if person_id is not None and e.get("person_id") not in (None, person_id):
            continue
        out.append(dict(e))
        if len(out) >= limit:
            break
    return out


# ── Action outcomes ──────────────────────────────────────────────────────────

def note_action_issued(seq: Optional[int], verb: str, detail: str = "", *,
                       requested_deg: Optional[float] = None,
                       attempted_deg: Optional[float] = None,
                       alternative: str = "") -> ActionResult:
    """A body command was accepted by the host and sent (status 'running').
    Returns the ActionResult so the issuer can annotate it further."""
    rec = ActionResult(
        verb=str(verb or "move"),
        detail=" ".join(str(detail or "").split()),
        seq=seq,
        requested_deg=requested_deg,
        attempted_deg=attempted_deg,
        alternative=" ".join(str(alternative or "").split()),
    )
    with _lock:
        _actions.append(rec)
    return rec


def note_action_result(seq: Optional[int], result: str, *, reason: str = "") -> None:
    """The firmware reported how a sent command ended (completed / blocked /
    aborted / superseded ...) or the host gave up on it (timeout / cancelled).
    Unknown seqs are ignored."""
    if seq is None:
        return
    with _lock:
        for rec in reversed(_actions):
            if rec.seq == seq:
                rec.finish(result, reason=reason)
                return


def note_action_verified(seq: Optional[int], *, requested_deg: float,
                         measured_deg: float) -> None:
    """The compass check measured how far a completed turn actually rotated.
    A mismatch beyond tolerance marks the record 'partial' — done, but not the
    heading that was asked for."""
    if seq is None:
        return
    with _lock:
        for rec in reversed(_actions):
            if rec.seq == seq:
                rec.measured_deg = float(measured_deg)
                if rec.requested_deg is None:
                    rec.requested_deg = float(requested_deg)
                tol = float(_cfg("MOTION_COMPASS_TURN_TOLERANCE_DEG", 4.0))
                if rec.status == "completed" and abs(float(requested_deg) - float(measured_deg)) > tol:
                    rec.status = "partial"
                return


def note_action_refused(verb: str, reason: str, detail: str = "", *,
                        requested_deg: Optional[float] = None) -> ActionResult:
    """The host refused to send a command (swing check, sensor fault, charging,
    manual override...). The reason is the code's, not the model's."""
    rec = ActionResult(verb=str(verb or "move"),
                       detail=" ".join(str(detail or "").split()),
                       seq=None, status="refused", reason=str(reason or ""),
                       requested_deg=requested_deg)
    rec.ended_at = time.monotonic()
    with _lock:
        _actions.append(rec)
    return rec


def recent_actions(*, max_age_secs: Optional[float] = None, limit: int = 3) -> list[dict]:
    max_age = float(max_age_secs if max_age_secs is not None
                    else _cfg("CONVERSATION_STATE_ACTION_TTL_SECS", 90.0))
    now = time.monotonic()
    with _lock:
        items = list(_actions)
    out = []
    for rec in reversed(items):
        if now - rec.at > max_age:
            continue
        out.append(rec.as_dict())
        if len(out) >= limit:
            break
    return out


def last_action_for_seq(seq: Optional[int]) -> Optional[dict]:
    if seq is None:
        return None
    with _lock:
        for rec in reversed(_actions):
            if rec.seq == seq:
                return rec.as_dict()
    return None


# ── Speaker resolution (phase 2B) ────────────────────────────────────────────

def note_speaker_resolution(resolution: Optional[dict]) -> None:
    """The attribution resolver's verdict for the CURRENT turn (see
    intelligence/attribution.py): {"status": known|unknown|ambiguous, "person_id",
    "name", "conflicts": [...], "note": str}. Replaces the previous turn's."""
    global _speaker_resolution
    with _lock:
        _speaker_resolution = dict(resolution) if resolution else None


def speaker_resolution() -> Optional[dict]:
    with _lock:
        return dict(_speaker_resolution) if _speaker_resolution else None


def speaker_lines() -> list[str]:
    res = speaker_resolution()
    if not res:
        return []
    status = str(res.get("status") or "")
    name = str(res.get("name") or "").split()[0] if res.get("name") else ""
    if status == "ambiguous":
        why = "; ".join(str(c) for c in (res.get("conflicts") or [])[:2])
        return [
            "SPEAKER UNCERTAIN: you are not sure who is talking right now"
            + (f" (best guess {name})" if name else "")
            + (f" — {why}" if why else "")
            + ". Answer them, but do NOT address them by name, do not treat stored personal "
            "memories as theirs, and do not file anything personal from this turn. If you "
            "genuinely need to know who it is, ONE short natural check ('is that you, "
            + (name or "friend") + "?') is fine — not on every line."
        ]
    if status == "unknown":
        return [
            "The current speaker is NOT someone you recognize (no known voice or face). "
            "Talk to them as a new person; don't guess a name."
        ]
    return []


_REASON_WORDS = {
    "swing_blocked": "a swing check said the body or arms would sweep into something",
    "not_connected": "the drive base is not connected",
    "robot_asleep": "you were asleep",
    "charging": "you are plugged into the charger",
    "interaction_paused": "interaction is paused",
    "manual_override": "someone has manual control of the base",
    "blocked": "an obstacle stopped it",
    "aborted": "the firmware aborted it",
    "timeout": "it never reported finishing",
    "cancelled": "it was cancelled",
    "suppressed": "the host would not send it",
    "not_settled": "the base did not settle afterwards",
}


def _reason_phrase(code: str) -> str:
    code = str(code or "").strip().lower()
    if not code:
        return ""
    if code.startswith("tof_"):
        return "your depth sensing is faulted, so you would not drive blind"
    return _REASON_WORDS.get(code, code.replace("_", " "))


def _age(secs: float) -> str:
    secs = max(0.0, float(secs))
    if secs < 5:
        return "just now"
    if secs < 90:
        return f"{int(secs)}s ago"
    return f"{int(secs // 60)} min ago"


# ── Rendering ────────────────────────────────────────────────────────────────

def pending_question_lines(current_person_id: Optional[int]) -> list[str]:
    """Rex's own unanswered questions, from the dialogue-act frames. Only frames
    aimed at someone OTHER than the current speaker (or at nobody in particular)
    are rendered — a frame aimed at the current speaker is answered by the very
    message the model is about to read."""
    try:
        from intelligence import dialogue_act
        frames = list(getattr(dialogue_act, "_frames", []) or [])
    except Exception:
        return []
    now = time.monotonic()
    lines: list[str] = []
    for frame in reversed(frames):
        try:
            if not frame.active(now) or "?" not in (frame.text or ""):
                continue
            target = frame.target_person_id
            if target is not None and current_person_id is not None \
                    and int(target) == int(current_person_id):
                continue
            who = frame.target_name
            if not who and target is not None:
                try:
                    from memory import people
                    p = people.get_person(int(target)) or {}
                    who = str(p.get("name") or "").split()[0] if p.get("name") else None
                except Exception:
                    who = None
            age = _age(now - frame.created_at)
            if who:
                lines.append(
                    f"You asked {who} {age}: \"{frame.text}\" — still unanswered by THEM. "
                    f"Someone else replying does not answer it for {who}."
                )
            else:
                lines.append(f"You asked {age}: \"{frame.text}\" — still unanswered.")
        except Exception:
            continue
        if len(lines) >= 2:
            break
    return lines


def render_lines(current_person_id: Optional[int]) -> list[str]:
    """Plain prompt lines for the current conversation's deterministic facts."""
    if not bool(_cfg("LEAN_CONTEXT_STATE_ENABLED", True)):
        return []
    out: list[str] = []
    corrections = recent_corrections(person_id=None)
    if corrections:
        out.append(
            "CORRECTIONS they made this conversation — these OVERRIDE anything older you "
            "remember or summarized, and you already acted on them (don't re-ask, don't "
            "apologize again): " + " | ".join(
                f"{c['text']} ({_age(time.monotonic() - c['at'])})" for c in corrections
            )
        )
    actions = recent_actions()
    if actions:
        parts = []
        now = time.monotonic()
        for a in actions:
            what = a["verb"] + (f" {a['detail']}" if a.get("detail") else "")
            if a.get("alternative"):
                what += f" ({a['alternative']})"
            elif a.get("shrunk"):
                what += (f" (asked {abs(a['requested_deg']):.0f}°, only {abs(a['attempted_deg']):.0f}° "
                         "was clear so that is what was sent)")
            status = a.get("status") or "running"
            if status == "refused":
                phrase = f"{what} → you REFUSED it ({_reason_phrase(a.get('reason'))})"
            elif status == "running":
                phrase = f"{what} → still in progress"
            elif status == "completed":
                phrase = f"{what} → done"
                if a.get("measured_deg") is not None:
                    phrase += f" (compass measured {abs(a['measured_deg']):.0f}°)"
            elif status == "partial":
                phrase = (f"{what} → finished but landed short/long: compass measured "
                          f"{abs(a.get('measured_deg') or 0):.0f}° of the "
                          f"{abs(a.get('requested_deg') or 0):.0f}° asked")
            else:
                phrase = f"{what} → ended '{status}' ({_reason_phrase(status)})"
            parts.append(f"{phrase}, {_age(now - a['at'])}")
        out.append(
            "What your BODY actually did recently (the record, not a guess — never claim a "
            "move happened if it says refused or blocked): " + "; ".join(parts)
        )
    out.extend(pending_question_lines(current_person_id))
    out.extend(speaker_lines())
    return out
