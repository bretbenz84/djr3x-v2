"""
tools/memory_cleanup.py — purge stored memories the quality gate now rejects.

The fact_quality gate stops NEW junk at the door, but the field DBs already hold
shards saved before each gate existed ('d the movie', 'for a job', 'lot', 'your
program to improve you'). This sweeps person_interests and person_facts through
the CURRENT gates and deletes what they reject — the same predicates, so the
sweep can only remove what the door would now refuse.

Usage:
    python tools/memory_cleanup.py            # dry run — show what WOULD go
    python tools/memory_cleanup.py --apply    # actually delete
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from memory import database as db, fact_quality  # noqa: E402

# Fact categories the interest-shard gates apply to (mirrors reject_fact scope).
_INTEREST_FACT_CATEGORIES = {"interest", "interest_note"}


def sweep(apply: bool) -> int:
    removed = 0

    rows = db.fetchall("SELECT id, person_id, name, notes FROM person_interests")
    for r in rows:
        row = dict(r)
        reason = fact_quality.reject_interest(
            str(row.get("name") or ""), str(row.get("notes") or "")
        )
        if not reason:
            continue
        removed += 1
        print(f"interest #{row['id']} (person {row['person_id']}): "
              f"{row['name']!r} — {reason}")
        if apply:
            db.execute("DELETE FROM person_interests WHERE id = ?", (row["id"],))

    rows = db.fetchall("SELECT id, person_id, category, key, value FROM person_facts")
    for r in rows:
        row = dict(r)
        category = str(row.get("category") or "")
        value = str(row.get("value") or "")
        reason = fact_quality.reject_fact(category, str(row.get("key") or ""), value)
        if not reason and category.strip().lower() in _INTEREST_FACT_CATEGORIES:
            reason = fact_quality.is_dangling_fragment(value)
        if not reason:
            continue
        removed += 1
        print(f"fact #{row['id']} (person {row['person_id']}): "
              f"{row['category']}/{row['key']} = {value!r} — {reason}")
        if apply:
            db.execute("DELETE FROM person_facts WHERE id = ?", (row["id"],))

    return removed


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--apply", action="store_true",
                        help="delete the rejected rows (default: dry run)")
    args = parser.parse_args()
    n = sweep(apply=args.apply)
    verb = "deleted" if args.apply else "would delete"
    print(f"\n{verb}: {n} row(s)")
    if n and not args.apply:
        print("re-run with --apply to delete")


if __name__ == "__main__":
    main()
