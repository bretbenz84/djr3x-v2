"""
memory/disposition.py — Long-running facial-expression disposition memory.

MediaPipe produces moment-to-moment expression reads. This module turns those
reads into per-person trends Rex can remember between runs.
"""

import logging
import sqlite3
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

_PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

import config
from memory import database as db

_log = logging.getLogger(__name__)

_TRACKED = ("smile", "frown", "neutral", "surprise", "brow_furrow")
_ALL_BUCKETS = (*_TRACKED, "other")
_SAMPLE_COLUMNS = {
    "smile": "smile_samples",
    "frown": "frown_samples",
    "neutral": "neutral_samples",
    "surprise": "surprise_samples",
    "brow_furrow": "brow_furrow_samples",
    "other": "other_samples",
}
_SCORE_COLUMNS = {
    "smile": "smile_score",
    "frown": "frown_score",
    "neutral": "neutral_score",
    "surprise": "surprise_score",
    "brow_furrow": "brow_furrow_score",
}
_ALIASES = {
    "happy": "smile",
    "joy": "smile",
    "joyful": "smile",
    "amused": "smile",
    "smiling": "smile",
    "sad": "frown",
    "unhappy": "frown",
    "downturned_mouth": "frown",
    "surprised": "surprise",
    "wide_eyed": "surprise",
    "wide_eyes": "surprise",
    "shocked": "surprise",
    "angry": "brow_furrow",
    "furrowed_brow": "brow_furrow",
    "irritated": "brow_furrow",
}
_PROMPT_PHRASES = {
    "smiley": "often visibly smiley",
    "grumpy": "often visibly frowny or displeased-looking",
    "deadpan": "usually neutral or deadpan",
    "intense": "often brow-furrowed or intensely focused-looking",
    "startled": "often surprised or wide-eyed-looking",
}


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _clean_label(value) -> str:
    return str(value or "").strip().lower().replace("-", "_").replace(" ", "_")


def canonical_expression(expression=None, mood=None) -> str:
    """Return one durable expression bucket for a MediaPipe/GUI expression read."""
    for candidate in (_clean_label(expression), _clean_label(mood)):
        if not candidate:
            continue
        candidate = _ALIASES.get(candidate, candidate)
        if candidate in _ALL_BUCKETS:
            return candidate
    return "other"


def _clamp(value, default: float = 0.0) -> float:
    try:
        return max(0.0, min(1.0, float(value)))
    except (TypeError, ValueError):
        return default


def _rolling_alpha() -> float:
    return _clamp(getattr(config, "FACIAL_DISPOSITION_ROLLING_ALPHA", 0.06), 0.06)


def _label_for(dominant: str, score: float) -> str:
    if dominant == "smile" and score >= 0.42:
        return "smiley"
    if dominant == "frown" and score >= 0.34:
        return "grumpy"
    if dominant == "neutral" and score >= 0.55:
        return "deadpan"
    if dominant == "brow_furrow" and score >= 0.34:
        return "intense"
    if dominant == "surprise" and score >= 0.30:
        return "startled"
    return "mixed"


def _dominant_from_scores(scores: dict[str, float]) -> tuple[str, float, str]:
    dominant = max(_TRACKED, key=lambda key: float(scores.get(key, 0.0) or 0.0))
    score = _clamp(scores.get(dominant))
    return dominant, score, _label_for(dominant, score)


def _row_to_dict(row) -> Optional[dict]:
    return dict(row) if row is not None else None


def get_stats(person_id: int) -> Optional[dict]:
    try:
        with db.connection() as conn:
            row = conn.execute(
                "SELECT * FROM person_disposition_stats WHERE person_id = ?",
                (int(person_id),),
            ).fetchone()
            return _row_to_dict(row)
    except sqlite3.OperationalError as exc:
        if "no such table" not in str(exc).lower():
            _log.debug("disposition stats lookup failed person_id=%s: %s", person_id, exc)
        return None
    except Exception as exc:
        _log.debug("disposition stats lookup failed person_id=%s: %s", person_id, exc)
        return None


def record_expression_sample(
    person_id: int,
    *,
    expression=None,
    mood=None,
    confidence: float = 0.0,
    observed_at: Optional[str] = None,
) -> Optional[dict]:
    """Fold one visible expression reading into durable per-person statistics."""
    try:
        pid = int(person_id)
    except (TypeError, ValueError):
        return None

    bucket = canonical_expression(expression, mood)
    now = observed_at or _now()
    confidence = _clamp(confidence)
    alpha = _rolling_alpha()

    try:
        with db.connection() as conn:
            row = conn.execute(
                "SELECT * FROM person_disposition_stats WHERE person_id = ?",
                (pid,),
            ).fetchone()
            if row is None:
                conn.execute(
                    """
                    INSERT INTO person_disposition_stats
                        (person_id, first_observed_at, last_observed_at)
                    VALUES (?, ?, ?)
                    """,
                    (pid, now, now),
                )
                row = conn.execute(
                    "SELECT * FROM person_disposition_stats WHERE person_id = ?",
                    (pid,),
                ).fetchone()

            current = dict(row)
            total = int(current.get("total_samples") or 0) + 1
            counts = {
                key: int(current.get(column) or 0)
                for key, column in _SAMPLE_COLUMNS.items()
            }
            counts[bucket] = counts.get(bucket, 0) + 1
            old_scores = {
                key: _clamp(current.get(column))
                for key, column in _SCORE_COLUMNS.items()
            }
            if total == 1:
                scores = {key: 1.0 if key == bucket else 0.0 for key in _TRACKED}
            else:
                scores = {
                    key: (old_scores[key] * (1.0 - alpha)) + (alpha if key == bucket else 0.0)
                    for key in _TRACKED
                }
            dominant, dominant_score, disposition_label = _dominant_from_scores(scores)
            conn.execute(
                """
                UPDATE person_disposition_stats
                   SET total_samples = ?,
                       smile_samples = ?,
                       frown_samples = ?,
                       neutral_samples = ?,
                       surprise_samples = ?,
                       brow_furrow_samples = ?,
                       other_samples = ?,
                       smile_score = ?,
                       frown_score = ?,
                       neutral_score = ?,
                       surprise_score = ?,
                       brow_furrow_score = ?,
                       dominant_expression = ?,
                       disposition_label = ?,
                       confidence = ?,
                       first_observed_at = COALESCE(first_observed_at, ?),
                       last_observed_at = ?
                 WHERE person_id = ?
                """,
                (
                    total,
                    counts["smile"],
                    counts["frown"],
                    counts["neutral"],
                    counts["surprise"],
                    counts["brow_furrow"],
                    counts["other"],
                    scores["smile"],
                    scores["frown"],
                    scores["neutral"],
                    scores["surprise"],
                    scores["brow_furrow"],
                    dominant,
                    disposition_label,
                    max(dominant_score, confidence if bucket == dominant else 0.0),
                    now,
                    now,
                    pid,
                ),
            )
            updated = conn.execute(
                "SELECT * FROM person_disposition_stats WHERE person_id = ?",
                (pid,),
            ).fetchone()
            return dict(updated) if updated else None
    except Exception as exc:
        _log.debug("disposition sample write failed person_id=%s: %s", person_id, exc)
        return None


def mark_mentioned(person_id: int, when: Optional[str] = None) -> None:
    try:
        pid = int(person_id)
    except (TypeError, ValueError):
        return
    db.execute(
        """
        UPDATE person_disposition_stats
           SET last_mentioned_at = ?
         WHERE person_id = ?
        """,
        (when or _now(), pid),
    )


def summarize_for_prompt(person_id: int, *, min_samples: int = 20) -> str:
    stats = get_stats(person_id)
    if not stats:
        return ""
    try:
        total = int(stats.get("total_samples") or 0)
    except (TypeError, ValueError):
        total = 0
    if total < max(1, int(min_samples)):
        return ""
    label = str(stats.get("disposition_label") or "").strip().lower()
    if label not in _PROMPT_PHRASES:
        return ""
    confidence = _clamp(stats.get("confidence"))
    phrase = _PROMPT_PHRASES[label]
    dominant = str(stats.get("dominant_expression") or "unknown").strip() or "unknown"
    return (
        f"Facial disposition trend: over {total} local expression samples, this person "
        f"is {phrase} (dominant visible expression: {dominant}, confidence {confidence:.2f}). "
        "Treat this as a light visual habit, not a diagnosis of inner emotion; use sparingly."
    )


def delete_stats(person_id: int) -> None:
    try:
        pid = int(person_id)
    except (TypeError, ValueError):
        return
    db.execute("DELETE FROM person_disposition_stats WHERE person_id = ?", (pid,))
