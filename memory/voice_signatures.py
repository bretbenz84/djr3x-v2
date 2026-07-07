"""
Cross-session memory for recurring UNKNOWN voices (voice-primary identity).

A *voice signature* is one persisted, L2-normalized voice embedding that Rex has
heard but has no name for yet. It lets him recognize a recurring voice across
sessions ("I've heard you before") and, the moment that voice is finally named,
link its accumulated samples to the new person — without ever creating a nameless
person row (which would leak into greetings, memory injection, etc.).

Lifecycle:
  - An anonymous session slot (``unknown_voice_N``) persists/refreshes a signature
    as it recurs (``record`` / ``bump``).
  - On a fresh unknown voice, ``match`` checks whether it resembles a signature
    seen in a previous session — that's the cross-session continuity hook.
  - When the voice is finally identified (off-screen identify / self-intro),
    ``attach_person`` links the signature to the person and its embedding is
    enrolled as a real voice biometric by the caller.

All reads/writes degrade gracefully if the ``voice_signatures`` table is missing
(older DB) and are gated by ``VOICE_SIGNATURE_PERSIST_ENABLED``.
"""

from __future__ import annotations

import logging
import os
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

import numpy as np

import config
from audio import voice_score as _voice_score
from memory import database as db

_log = logging.getLogger("memory.voice_signatures")

_TABLE = "voice_signatures"
_table_checked = False
_table_present = False


def enabled() -> bool:
    return bool(getattr(config, "VOICE_SIGNATURE_PERSIST_ENABLED", True))


def _under_test_runner() -> bool:
    if os.environ.get("PYTEST_CURRENT_TEST"):
        return True
    argv0 = (sys.argv[0] if sys.argv else "").lower()
    return "unittest" in argv0 or "pytest" in argv0 or "py.test" in argv0


def _writes_suppressed() -> bool:
    """Under the test runner, never write the REAL people.db. A test that points
    ``database._DB_FILE`` at a temp file (the standard fixture) opts back in, since
    its path no longer equals the default."""
    if not _under_test_runner():
        return False
    try:
        default = Path(getattr(db, "_PROJECT_ROOT")) / config.DB_PATH
        return Path(db._DB_FILE) == default
    except Exception:
        return True


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _table_available() -> bool:
    """True once, cached, if the voice_signatures table exists (older DBs lack it)."""
    global _table_checked, _table_present
    if _table_checked:
        return _table_present
    try:
        row = db.fetchone(
            "SELECT name FROM sqlite_master WHERE type='table' AND name=?",
            (_TABLE,),
        )
        _table_present = row is not None
    except Exception as exc:
        _log.debug("voice_signatures table check failed: %s", exc)
        _table_present = False
    _table_checked = True
    return _table_present


def _to_blob(embedding: np.ndarray) -> bytes:
    return np.asarray(embedding, dtype=np.float32).tobytes()


def _from_blob(blob: bytes) -> np.ndarray:
    return np.frombuffer(blob, dtype=np.float32)


def _normalize(embedding) -> Optional[np.ndarray]:
    try:
        arr = np.asarray(embedding, dtype=np.float32).reshape(-1)
    except Exception:
        return None
    if arr.size == 0:
        return None
    norm = float(np.linalg.norm(arr))
    if norm <= 1e-10:
        return None
    return (arr / norm).astype(np.float32)


def match(embedding) -> Optional[dict]:
    """Return the best-matching signature for this embedding, or None.

    Result: ``{"id", "score", "person_id", "turns", "label"}``. ``score`` is cosine
    similarity (embeddings are L2-normalized). Only returns a match at or above
    ``VOICE_SIGNATURE_MATCH_THRESHOLD``.
    """
    if not enabled() or not _table_available():
        return None
    query = _normalize(embedding)
    if query is None:
        return None
    threshold = float(getattr(config, "VOICE_SIGNATURE_MATCH_THRESHOLD", 0.74))
    try:
        rows = db.fetchall(
            "SELECT id, embedding, turns, person_id, label, last_seen_at FROM voice_signatures"
        )
    except Exception as exc:
        _log.debug("voice_signatures match query failed: %s", exc)
        return None
    best: Optional[dict] = None
    for row in rows:
        vec = _normalize(_from_blob(row["embedding"]))
        if vec is None or vec.shape != query.shape:
            continue
        # Mapped onto the Resemblyzer-calibrated threshold scale (see voice_score).
        score = _voice_score.map_similarity(float(np.dot(vec, query)))
        if best is None or score > best["score"]:
            best = {
                "id": int(row["id"]),
                "score": score,
                "person_id": row["person_id"],
                "turns": int(row["turns"] or 0),
                "label": row["label"],
                "last_seen_at": row["last_seen_at"],
            }
    if best is None or best["score"] < threshold:
        return None
    return best


def record(embedding, *, label: Optional[str] = None) -> Optional[int]:
    """Persist a brand-new signature for an unknown voice. Returns its id."""
    if not enabled() or _writes_suppressed() or not _table_available():
        return None
    vec = _normalize(embedding)
    if vec is None:
        return None
    now = _now()
    try:
        return db.execute(
            "INSERT INTO voice_signatures (embedding, turns, label, created_at, last_seen_at) "
            "VALUES (?, 1, ?, ?, ?)",
            (_to_blob(vec), label, now, now),
        )
    except Exception as exc:
        _log.debug("voice_signatures record failed: %s", exc)
        return None


def bump(signature_id: int, embedding) -> None:
    """Blend the new sample into an existing signature and increment its turn count."""
    if not enabled() or _writes_suppressed() or not _table_available() or signature_id is None:
        return
    vec = _normalize(embedding)
    if vec is None:
        return
    try:
        row = db.fetchone(
            "SELECT embedding, turns FROM voice_signatures WHERE id=?",
            (int(signature_id),),
        )
        if row is None:
            return
        prior = _normalize(_from_blob(row["embedding"]))
        turns = int(row["turns"] or 1)
        if prior is not None and prior.shape == vec.shape:
            # Weighted running mean, capped so old samples don't ossify the print.
            weight = float(min(max(turns, 1), 5))
            blended = _normalize((prior * weight) + vec)
        else:
            blended = vec
        db.execute(
            "UPDATE voice_signatures SET embedding=?, turns=turns+1, last_seen_at=? WHERE id=?",
            (_to_blob(blended if blended is not None else vec), _now(), int(signature_id)),
        )
    except Exception as exc:
        _log.debug("voice_signatures bump failed: %s", exc)


def attach_person(signature_id: Optional[int], person_id: int) -> None:
    """Link a signature to a now-known person (promotion). Idempotent."""
    if (
        not enabled()
        or _writes_suppressed()
        or not _table_available()
        or signature_id is None
        or person_id is None
    ):
        return
    try:
        db.execute(
            "UPDATE voice_signatures SET person_id=?, last_seen_at=? WHERE id=?",
            (int(person_id), _now(), int(signature_id)),
        )
        _log.info(
            "[voice_signatures] linked signature id=%s to person_id=%s",
            signature_id, person_id,
        )
    except Exception as exc:
        _log.debug("voice_signatures attach_person failed: %s", exc)


def reset_table_cache() -> None:
    """Test hook: re-check table presence after a DB swap."""
    global _table_checked, _table_present
    _table_checked = False
    _table_present = False
