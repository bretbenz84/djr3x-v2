"""
memory/social.py — Inter-person relationship graph.

Stores directional edges between people in the person_relationships table:
"Bret says JT is his partner" → (from=Bret, to=JT, relationship="partner", by=Bret).

Symmetric relationships (partner, spouse, friend, sibling, etc.) are auto-mirrored
so a lookup on either person surfaces the edge. Asymmetric labels (son, boss,
employee, parent, child, etc.) are stored as-is without a mirror.
"""

import logging
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

_PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from memory import database as db

_log = logging.getLogger(__name__)


# Labels that are symmetric — if A→B is "partner" then B→A is also "partner".
_SYMMETRIC_LABELS = frozenset({
    "partner", "spouse", "wife", "husband", "girlfriend", "boyfriend",
    "friend", "bestfriend", "best_friend",
    "sibling", "brother", "sister",
    "cousin",
    "roommate", "flatmate",
    "neighbor", "neighbour",
    "colleague", "coworker", "co_worker",
    "classmate",
})

# Directional labels are stored from the speaker's perspective:
# Bret -> Bob, "father" means "Bob is Bret's father."
_DIRECTIONAL_DISPLAY = {
    "father": "{to} is {from_}'s father",
    "mother": "{to} is {from_}'s mother",
    "parent": "{to} is {from_}'s parent",
    "aunt": "{to} is {from_}'s aunt",
    "uncle": "{to} is {from_}'s uncle",
    "boss": "{to} is {from_}'s boss",
    "supervisor": "{to} is {from_}'s supervisor",
    "manager": "{to} is {from_}'s manager",
    "employee": "{to} is {from_}'s employee",
    "child": "{to} is {from_}'s child",
    "son": "{to} is {from_}'s son",
    "daughter": "{to} is {from_}'s daughter",
    "owner": "{to} is {from_}'s owner",
}


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _normalize(label: str) -> str:
    return (label or "").strip().lower().replace("-", "_").replace(" ", "_")


def save_relationship(
    from_person_id: int,
    to_person_id: int,
    relationship: str,
    described_by: Optional[int] = None,
) -> None:
    """Store (or upsert) a relationship edge. Mirrors symmetric labels automatically."""
    if from_person_id is None or to_person_id is None or not relationship:
        return
    if from_person_id == to_person_id:
        return

    label = _normalize(relationship)
    now = _now()

    def _upsert(a: int, b: int, lbl: str, by: Optional[int]) -> None:
        existing = db.fetchone(
            """SELECT id FROM person_relationships
               WHERE from_person_id = ? AND to_person_id = ? AND relationship = ?""",
            (a, b, lbl),
        )
        if existing:
            db.execute(
                "UPDATE person_relationships SET described_by = ?, updated_at = ? WHERE id = ?",
                (by, now, existing["id"]),
            )
        else:
            db.execute(
                """INSERT INTO person_relationships
                   (from_person_id, to_person_id, relationship, described_by, created_at, updated_at)
                   VALUES (?, ?, ?, ?, ?, ?)""",
                (a, b, lbl, by, now, now),
            )

    _upsert(from_person_id, to_person_id, label, described_by)
    if label in _SYMMETRIC_LABELS:
        _upsert(to_person_id, from_person_id, label, described_by)

    _log.info(
        "social: saved edge person_id=%s --[%s]--> person_id=%s (by=%s, mirror=%s)",
        from_person_id, label, to_person_id, described_by,
        label in _SYMMETRIC_LABELS,
    )


def get_outbound(person_id: int) -> list[dict]:
    """Edges FROM this person toward others."""
    rows = db.fetchall(
        """SELECT r.*, p.name AS to_name FROM person_relationships r
           JOIN people p ON p.id = r.to_person_id
           WHERE r.from_person_id = ?
           ORDER BY r.updated_at DESC""",
        (person_id,),
    )
    return [dict(r) for r in rows]


def get_inbound(person_id: int) -> list[dict]:
    """Edges FROM others TOWARD this person."""
    rows = db.fetchall(
        """SELECT r.*, p.name AS from_name FROM person_relationships r
           JOIN people p ON p.id = r.from_person_id
           WHERE r.to_person_id = ?
           ORDER BY r.updated_at DESC""",
        (person_id,),
    )
    return [dict(r) for r in rows]


def get_all_involving(person_id: int) -> list[dict]:
    """All edges where person_id is either end. Deduplicated by edge row id."""
    out = get_outbound(person_id)
    seen = {r["id"] for r in out}
    merged = list(out)
    for row in get_inbound(person_id):
        if row["id"] not in seen:
            merged.append(row)
    return merged


def get_between(person_a_id: int, person_b_id: int) -> list[dict]:
    """Return relationship edges directly connecting two people in either direction."""
    rows = db.fetchall(
        """SELECT r.*,
                  pf.name AS from_name,
                  pt.name AS to_name
           FROM person_relationships r
           JOIN people pf ON pf.id = r.from_person_id
           JOIN people pt ON pt.id = r.to_person_id
           WHERE (r.from_person_id = ? AND r.to_person_id = ?)
              OR (r.from_person_id = ? AND r.to_person_id = ?)
           ORDER BY r.updated_at DESC""",
        (int(person_a_id), int(person_b_id), int(person_b_id), int(person_a_id)),
    )
    return [dict(r) for r in rows]


def delete_for_person(person_id: int) -> None:
    """Remove all relationship edges involving person_id (both directions)."""
    db.execute(
        "DELETE FROM person_relationships WHERE from_person_id = ? OR to_person_id = ?",
        (person_id, person_id),
    )


def summarize_for_prompt(person_id: int, person_name: str) -> str:
    """
    Return a short human-readable summary of person_id's relationships for
    injection into the LLM system prompt. Empty string if none.
    """
    edges = get_all_involving(person_id)
    if not edges:
        return ""
    parts: list[str] = []
    for edge in edges:
        rel = edge["relationship"]
        if edge["from_person_id"] == person_id:
            other = edge.get("to_name") or f"person_{edge['to_person_id']}"
            if rel in _DIRECTIONAL_DISPLAY:
                parts.append(_DIRECTIONAL_DISPLAY[rel].format(from_=person_name, to=other))
            else:
                parts.append(f"{person_name} is {rel} of {other}")
        else:
            other = edge.get("from_name") or f"person_{edge['from_person_id']}"
            if rel in _DIRECTIONAL_DISPLAY:
                parts.append(_DIRECTIONAL_DISPLAY[rel].format(from_=other, to=person_name))
            else:
                parts.append(f"{other} says {person_name} is their {rel}")
    # Dedup identical lines (symmetric mirrors can produce duplicates once joined on names)
    seen: set[str] = set()
    unique = []
    for p in parts:
        if p not in seen:
            seen.add(p)
            unique.append(p)
    return "; ".join(unique)
