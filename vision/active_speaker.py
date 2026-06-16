"""
Active-speaker detection (visual) — vision/active_speaker.py

When two or more people are in frame, Rex needs to know WHICH visible person is
currently speaking so conversation attribution, gaze/face-tracking, and memory
target the right individual. Audio direction-of-arrival is unavailable (the
ReSpeaker Lite delivers a single AEC stream), so this is solved VISUALLY: per-face
lip-motion energy (jawOpen variance), gated on head orientation (yaw) and the live
VAD "is human speech happening right now" flag, arbitrated into a single winner.
The result is published as a per-person ``is_speaking`` signal on
``world_state.people``. Full design: docs/active_speaker_detection.md.

It piggybacks on the Face Landmarker data ``vision/face_expression.py`` already
computes every cycle, and reuses that module's IoU face→person association — so it
adds buffering + math, not inference, and never re-detects or re-associates faces.

Build status — COMMIT 1 (scaffold): the world-state write/latch/read CONTRACT is
implemented and tested here (it is the race-prone part). The detection layers —
1: head-pose gate, 2: lip-motion energy, 3: arbitration + hysteresis — land in
subsequent commits. ``update()`` is a no-op until then, so Rex behaves exactly as
before while this is built up.
"""

from __future__ import annotations

import logging
import threading
import time
from typing import Optional

import config

_log = logging.getLogger("vision.active_speaker")

# ── Module state (guarded by _lock, like the other vision modules) ─────────────
_lock = threading.Lock()

# Layer 2 (commit 2): per-person rolling jawOpen buffers.
#   key = ("pid", int) when identity is known, else ("slot", int) in-frame index.
_buffers: dict = {}

# Layer 3 (commit 5): arbitration / hysteresis state.
_current_speaker_key = None
_switch_candidate_key = None
_switch_since: float = 0.0
_last_vad_active_at: float = 0.0

# Latched "who was visually speaking most recently". Read by VOICE identity
# resolution AFTER a turn ends — when the live is_speaking has already been
# released — so vision can disambiguate WHICH visible person spoke without racing
# the real-time signal. Shape: {"person_db_id", "slot_idx", "confidence", "at"}.
_last_active: Optional[dict] = None


def _safe_int(value) -> Optional[int]:
    try:
        return int(value) if value is not None else None
    except (TypeError, ValueError):
        return None


def enabled() -> bool:
    return bool(getattr(config, "ACTIVE_SPEAKER_ENABLED", True))


# ── Entry point ────────────────────────────────────────────────────────────────

def update(face_signals: list[dict], vad_active: bool) -> None:
    """Run active-speaker detection for one face-expression cycle and publish the
    result. ``face_signals`` is a list of per-face dicts built by the
    face_expression hook (reusing the existing IoU match), each:
    ``{"slot_idx": int, "person_db_id": Optional[int], "jaw_open": float,
    "yaw": Optional[float], "ts": float}``.

    COMMIT 1: no-op stub. Layer 1 (head-pose gate), Layer 2 (lip-motion energy),
    and Layer 3 (arbitration + hysteresis) are added in commits 2–5; until then
    this writes nothing and the rest of Rex is unchanged.
    """
    if not enabled():
        return
    # TODO(commit 2): buffer jaw_open per key, compute lip_energy/lip_active.
    # TODO(commit 3): facing_camera gate from yaw.
    # TODO(commit 5): VAD gate, candidate set, margin winner, hysteresis, then
    #   _publish_speaker(winner_pid=..., winner_slot=..., confidence=...).
    return


# ── World-state write contract (implemented + tested in commit 1) ──────────────

def _write_speaker_fields(people, *, winner_pid, winner_slot, confidence, now):
    """Pure ``world_state.mutate("people", ...)`` callback: set ``is_speaking`` on
    exactly the winner slot — matched by ``person_db_id`` when available (stable
    across a slot resize), else by in-frame slot index — and clear it on every
    other slot. Exactly zero or one slot ends up True.

    Runs under the world_state lock, so it must be fast and must NOT call back into
    world_state (get/update/mutate would deadlock — see world_state.mutate docs).
    """
    win_pid = _safe_int(winner_pid)
    for i, slot in enumerate(people):
        if not isinstance(slot, dict):
            continue
        if win_pid is not None and slot.get("person_db_id") is not None:
            is_win = _safe_int(slot.get("person_db_id")) == win_pid
        else:
            is_win = winner_slot is not None and i == winner_slot
        slot["is_speaking"] = bool(is_win)
        if is_win:
            slot["speaking_confidence"] = float(confidence)
            slot["speaking_updated_at"] = now
    return people


def _publish_speaker(*, winner_pid, winner_slot, confidence, now=None) -> None:
    """Write the winner to world_state and refresh the recent-speaker latch.

    Passing both ``winner_pid`` and ``winner_slot`` as None clears the live
    ``is_speaking`` on all slots (no winner this cycle) WITHOUT touching the latch,
    so ``recent_visual_speaker`` keeps decaying naturally over the turn.
    """
    now = now if now is not None else time.time()
    try:
        from world_state import world_state
        world_state.mutate(
            "people",
            lambda people: _write_speaker_fields(
                people,
                winner_pid=winner_pid,
                winner_slot=winner_slot,
                confidence=confidence,
                now=now,
            ),
        )
    except Exception as exc:
        _log.debug("active_speaker world-state write failed: %s", exc)
        return
    if winner_pid is not None or winner_slot is not None:
        global _last_active
        with _lock:
            _last_active = {
                "person_db_id": _safe_int(winner_pid),
                "slot_idx": winner_slot,
                "confidence": float(confidence),
                "at": now,
            }


# ── Consumer helpers ───────────────────────────────────────────────────────────

def current_speaker(snapshot=None) -> Optional[dict]:
    """The visible person whose ``is_speaking`` is True and FRESH right now,
    resolved to a name — mirrors ``face.visible_known_names``. Returns the
    highest-confidence such slot as ``{person_db_id, name, speaking_confidence,
    speaking_updated_at}`` or None. For real-time consumers (e.g. face-tracking);
    voice attribution should use ``recent_visual_speaker`` instead. Never raises.
    """
    try:
        if snapshot is not None:
            entries = (snapshot or {}).get("people") or []
        else:
            from world_state import world_state
            entries = world_state.get("people") or []
    except Exception:
        return None

    stale = float(getattr(config, "ACTIVE_SPEAKER_STALE_SECS", 1.0))
    now = time.time()
    best: Optional[dict] = None
    for slot in entries:
        if not isinstance(slot, dict):
            continue
        if not slot.get("is_speaking"):
            continue
        if slot.get("face_visible") is False or slot.get("face_missing"):
            continue
        updated = slot.get("speaking_updated_at") or 0.0
        if now - float(updated) > stale:
            continue
        conf = float(slot.get("speaking_confidence") or 0.0)
        if best is None or conf > best["speaking_confidence"]:
            best = {
                "person_db_id": _safe_int(slot.get("person_db_id")),
                "name": None,
                "speaking_confidence": conf,
                "speaking_updated_at": float(updated),
            }
    if best is None:
        return None
    if best["person_db_id"] is not None:
        try:
            from memory import people as _people
            row = _people.get_person(best["person_db_id"])
            best["name"] = (row or {}).get("name")
        except Exception:
            best["name"] = None
    return best


def recent_visual_speaker(max_age_secs: Optional[float] = None) -> Optional[dict]:
    """The person who was visually speaking most recently, within ``max_age_secs``
    (default ``ACTIVE_SPEAKER_LATCH_SECS``).

    This is the signal VOICE identity resolution reads. Voice attribution runs
    AFTER a turn ends — past the silence timeout and transcription — by which point
    the live ``is_speaking`` slot field has already been released. The latch
    survives that gap, letting the voice tie-breaker use vision to disambiguate
    WHICH visible person spoke without racing the real-time signal. Returns
    ``{person_db_id, name, confidence, at}`` or None. Never raises.
    """
    if max_age_secs is None:
        max_age_secs = float(getattr(config, "ACTIVE_SPEAKER_LATCH_SECS", 3.0))
    with _lock:
        latched = dict(_last_active) if _last_active else None
    if not latched:
        return None
    if time.time() - float(latched.get("at") or 0.0) > float(max_age_secs):
        return None
    out = {
        "person_db_id": latched.get("person_db_id"),
        "name": None,
        "confidence": float(latched.get("confidence") or 0.0),
        "at": float(latched.get("at") or 0.0),
    }
    if out["person_db_id"] is not None:
        try:
            from memory import people as _people
            row = _people.get_person(out["person_db_id"])
            out["name"] = (row or {}).get("name")
        except Exception:
            out["name"] = None
    return out


def reset() -> None:
    """Drop all buffers + arbitration + latch state. Called from
    face_expression.stop() and by tests."""
    global _current_speaker_key, _switch_candidate_key, _switch_since
    global _last_vad_active_at, _last_active
    with _lock:
        _buffers.clear()
        _current_speaker_key = None
        _switch_candidate_key = None
        _switch_since = 0.0
        _last_vad_active_at = 0.0
        _last_active = None
