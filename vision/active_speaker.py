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
import math
import threading
import time
from collections import deque
from typing import Optional

import numpy as np

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

    Layer 2 (lip-motion energy), Layer 1 (head-pose gate), Layer 3 (VAD gate +
    margin winner + hysteresis) run here under ``_lock``; the world-state write +
    latch happen AFTER the lock is released (``_publish_speaker`` takes its own
    locks — calling it while holding ``_lock`` would deadlock).
    """
    if not enabled():
        return
    now = time.time()
    decision = _decide(face_signals, bool(vad_active), now)
    if decision is None:
        # No winner this cycle (VAD silent, no articulating mouth, or off-screen
        # speech): clear the live is_speaking on all slots; leave the latch to
        # decay. This empty result during speech is the correct off-screen signal.
        _publish_speaker(winner_pid=None, winner_slot=None, confidence=0.0, now=now)
    else:
        pid, slot, conf = decision
        _publish_speaker(winner_pid=pid, winner_slot=slot, confidence=conf, now=now)


def _decide(face_signals, vad_active, now):
    """Pure-ish arbitration over one cycle's per-face signals. Mutates the module
    buffers + hysteresis state under ``_lock`` and returns the winner as
    ``(person_db_id, slot_idx, confidence)`` or None (no live speaker). Separated
    from the world-state write so it can be unit-tested directly."""
    global _current_speaker_key, _switch_candidate_key, _switch_since, _last_vad_active_at

    window = float(getattr(config, "LIPSYNC_WINDOW_SECS", 1.0))
    stale = float(getattr(config, "LIPSYNC_STALE_SECS", 2.0))
    energy_threshold = float(getattr(config, "LIPSYNC_ENERGY_THRESHOLD", 0.0025))
    facing_max = float(getattr(config, "FACING_YAW_MAX_DEG", 30.0))
    margin = float(getattr(config, "SPEAKER_MARGIN", 0.0015))
    switch_margin = float(getattr(config, "SPEAKER_SWITCH_MARGIN", 0.0030))
    switch_secs = float(getattr(config, "SPEAKER_SWITCH_SECS", 0.4))
    release_secs = float(getattr(config, "SPEAKER_RELEASE_SECS", 0.6))

    with _lock:
        # ── Layer 3a: VAD gate. No human speech → nobody is the active speaker. ──
        if vad_active:
            _last_vad_active_at = now
        speaking_now = vad_active or (now - _last_vad_active_at) <= release_secs

        # ── Layer 2: per-person lip-motion energy from rolling jawOpen buffers. ──
        candidates: list[dict] = []
        seen_keys = set()
        for sig in face_signals or []:
            slot_idx = sig.get("slot_idx")
            pid = _safe_int(sig.get("person_db_id"))
            # Key by stable identity when known, else the in-frame slot index. No
            # mid-stream re-keying: if identity resolves, the old slot buffer ages
            # out and a fresh pid buffer fills within one window (~4 samples).
            key = ("pid", pid) if pid is not None else ("slot", slot_idx)
            seen_keys.add(key)
            ts = float(sig.get("ts") or now)
            jaw = float(sig.get("jaw_open") or 0.0)
            buf = _buffers.setdefault(key, deque())
            buf.append((ts, jaw))
            while buf and (ts - buf[0][0]) > window:
                buf.popleft()
            energy = _variance([v for (_, v) in buf])
            yaw = sig.get("yaw")
            facing = (yaw is None) or (abs(float(yaw)) <= facing_max)  # Layer 1 gate
            candidates.append({
                "key": key, "pid": pid, "slot_idx": slot_idx,
                "energy": energy, "lip_active": energy >= energy_threshold,
                "facing": facing,
            })

        # Age out buffers for faces not seen this cycle (leave/return).
        for key in list(_buffers.keys()):
            if key in seen_keys:
                continue
            buf = _buffers[key]
            if not buf or (now - buf[-1][0]) > stale:
                del _buffers[key]

        # ── Layer 3b: arbitration. ──
        by_key = {c["key"]: c for c in candidates}
        if not speaking_now:
            # No human speech: nobody is the active speaker. (Buffers above still
            # updated so energy history stays continuous for the next utterance.)
            _current_speaker_key = None
            _switch_candidate_key = None
            _switch_since = 0.0
            final_key = None
            result = None
        else:
            # Candidate set: faces turned toward Rex; fall back to all if that
            # empties (everyone slightly turned during real speech — still
            # attribute someone).
            facing_pool = [c for c in candidates if c["facing"]]
            pool = facing_pool if facing_pool else candidates
            active = [c for c in pool if c["lip_active"]]
            final_key = _arbitrate(active, now, margin, switch_margin, switch_secs)
            if final_key is not None and final_key in by_key:
                win = by_key[final_key]
                result = (win["pid"], win["slot_idx"], _confidence(win["energy"], energy_threshold))
            else:
                final_key = None
                result = None

        # Calibration scoreboard: log every cycle that has a visible face, so lip
        # energy is observable during BOTH speech and silence (off by default).
        if candidates and getattr(config, "ACTIVE_SPEAKER_LOG_SCOREBOARD", False):
            _log_scoreboard(candidates, final_key, vad_active=vad_active)
        return result


def _arbitrate(active, now, margin, switch_margin, switch_secs):
    """Pick the winning buffer key with hysteresis. Assumes ``_lock`` is held."""
    global _current_speaker_key, _switch_candidate_key, _switch_since

    active_sorted = sorted(active, key=lambda c: c["energy"], reverse=True)
    cur_key = _current_speaker_key
    cur = next((c for c in active_sorted if c["key"] == cur_key), None)

    if not active_sorted:
        # Nobody articulating this instant (e.g. a mid-sentence closed-mouth gap
        # while VAD is still on). Hold the current speaker through the gap; the
        # VAD release in _decide is what ultimately clears them.
        return cur_key

    top = active_sorted[0]
    runner = active_sorted[1]["energy"] if len(active_sorted) > 1 else None

    if cur is None:
        # No (articulating) current speaker — take the top only if it clearly
        # beats the runner-up; single articulating face wins outright (runner None).
        if runner is None or (top["energy"] - runner) >= margin:
            _current_speaker_key = top["key"]
            _switch_candidate_key = None
            _switch_since = 0.0
            return top["key"]
        return cur_key  # ambiguous pair — don't (re)assign; hold whatever we had

    if top["key"] == cur["key"]:
        _switch_candidate_key = None
        _switch_since = 0.0
        return cur["key"]

    # A challenger leads the current speaker — must beat them by the (higher)
    # switch margin, sustained for switch_secs, before stealing the floor.
    if (top["energy"] - cur["energy"]) >= switch_margin:
        if _switch_candidate_key == top["key"]:
            if (now - _switch_since) >= switch_secs:
                _current_speaker_key = top["key"]
                _switch_candidate_key = None
                _switch_since = 0.0
                return top["key"]
        else:
            _switch_candidate_key = top["key"]
            _switch_since = now
        return cur["key"]  # hold incumbent during the switch window

    _switch_candidate_key = None
    _switch_since = 0.0
    return cur["key"]


def _key_label(c: dict) -> str:
    kind, val = c["key"]
    return f"{'pid' if kind == 'pid' else 'slot'}:{val}"


def _log_scoreboard(candidates, final_key, *, vad_active: bool) -> None:
    """Calibration log, mirroring speaker_id._log_scoreboard. Off by default
    (ACTIVE_SPEAKER_LOG_SCOREBOARD); the calibration tool turns it on."""
    facing = ",".join(_key_label(c) for c in candidates if c["facing"]) or "-"
    energy = " ".join(
        f"{_key_label(c)}={c['energy']:.4f}{'*' if c['lip_active'] else ''}"
        for c in sorted(candidates, key=lambda c: c["energy"], reverse=True)
    ) or "-"
    win = "none"
    if final_key is not None:
        win = next((_key_label(c) for c in candidates if c["key"] == final_key), str(final_key))
    _log.info(
        "[active_speaker] vad=%s facing={%s} energy: %s -> speaking=%s",
        "on" if vad_active else "off", facing, energy, win,
    )


def _variance(values) -> float:
    n = len(values)
    if n < 2:
        return 0.0
    mean = sum(values) / n
    return sum((v - mean) ** 2 for v in values) / n


def _confidence(energy: float, threshold: float) -> float:
    """Map lip-motion energy to a 0..1 confidence. Placeholder scaling — the
    absolute energy scale is calibrated on-device (commit 6)."""
    if threshold <= 0:
        return 1.0 if energy > 0 else 0.0
    return max(0.0, min(1.0, energy / (threshold * 4.0)))


# ── Layer 1: head-pose yaw ─────────────────────────────────────────────────────

def yaw_from_transform_matrix(matrix) -> Optional[float]:
    """Head yaw in DEGREES from MediaPipe's 4×4 facial transformation matrix
    (free — already computed when ``output_facial_transformation_matrixes`` is on).
    Pure numpy, no cv2. 0° = facing the camera; sign/axis convention is confirmed
    on-device in commit 6 (the calibration tool logs this alongside the landmark
    fallback). Returns None if the matrix is missing/malformed.
    """
    if matrix is None:
        return None
    try:
        m = np.asarray(matrix, dtype=np.float64)
        if m.size < 12:
            return None
        m = m.reshape(4, 4) if m.size == 16 else m.reshape(m.shape)
        r = m[:3, :3]
        # Yaw about the vertical axis; verified to give ~30° for a 30° Y-rotation.
        return float(math.degrees(math.atan2(-r[2, 0], math.hypot(r[0, 0], r[1, 0]))))
    except Exception:
        return None


# MediaPipe FaceMesh canonical indices for the asymmetry fallback.
_NOSE_TIP = 1
_LEFT_EYE_OUTER = 33
_RIGHT_EYE_OUTER = 263


def _yaw_from_landmarks(landmarks) -> Optional[float]:
    """FALLBACK head-pose estimate (spec Option A): horizontal nose/eye asymmetry.

    NOT wired by default — the transformation-matrix method above is primary. This
    returns a NORMALIZED skew in roughly [-1, +1] (not degrees), so adopting it
    means re-tuning the facing gate against a normalized threshold. Kept so the
    calibration tool can log it next to the matrix yaw for comparison (commit 6).
    """
    try:
        nose = landmarks[_NOSE_TIP]
        le = landmarks[_LEFT_EYE_OUTER]
        re = landmarks[_RIGHT_EYE_OUTER]
        nx = float(getattr(nose, "x", nose[0] if not hasattr(nose, "x") else 0.0))
        lx = float(getattr(le, "x", le[0] if not hasattr(le, "x") else 0.0))
        rx = float(getattr(re, "x", re[0] if not hasattr(re, "x") else 0.0))
        interocular = abs(rx - lx)
        if interocular <= 1e-6:
            return None
        # (nose→left) vs (right→nose): symmetric ⇒ 0; skewed ⇒ turned.
        return float(((nx - lx) - (rx - nx)) / interocular)
    except Exception:
        return None


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
