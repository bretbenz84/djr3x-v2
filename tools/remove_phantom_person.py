"""Remove an ASR-poisoned phantom person row and every row that references it.

Usage:
    venv/bin/python tools/remove_phantom_person.py "Mm-hmm"
    venv/bin/python tools/remove_phantom_person.py --id 4

Created for the 2026-07-30 incident: Whisper transcribed a backchannel
("Mm-hmm") as an identity-prompt answer, enrolling Bret's own face AND voice
under a phantom person whose fresher voiceprint then outscored Bret on his own
speech (0.93-0.99 vs 0.43-0.79), mis-attributing turns and suppressing his
first-sight greeting. memory/name_validation.py now rejects backchannel names
at the source; this tool cleans up any row that slipped through before that.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import config  # noqa: E402
import sqlite3  # noqa: E402

# Every table with a person-reference column, checked dynamically so schema
# additions are covered without editing this list.
_PERSON_REF_COLUMNS = {"person_id", "person_a_id", "person_b_id", "subject_person_id"}


def remove_person(db_path: str, *, person_id: int | None, name: str | None) -> int:
    db = sqlite3.connect(db_path)
    try:
        cur = db.cursor()
        if person_id is None:
            row = cur.execute(
                "SELECT id, name FROM people WHERE name = ? COLLATE NOCASE", (name,)
            ).fetchone()
            if row is None:
                print(f"No person named {name!r} — nothing to do.")
                return 0
            person_id, name = int(row[0]), row[1]
        else:
            row = cur.execute(
                "SELECT name FROM people WHERE id = ?", (person_id,)
            ).fetchone()
            if row is None:
                print(f"No person id={person_id} — nothing to do.")
                return 0
            name = row[0]

        print(f"Removing person id={person_id} name={name!r} and all referencing rows:")
        tables = [r[0] for r in cur.execute("SELECT name FROM sqlite_master WHERE type='table'")]
        for table in tables:
            cols = {c[1] for c in cur.execute(f"PRAGMA table_info({table})")}
            for col in cols & _PERSON_REF_COLUMNS:
                cur.execute(f"DELETE FROM {table} WHERE {col} = ?", (person_id,))
                if cur.rowcount:
                    print(f"  {table}.{col}: {cur.rowcount} row(s)")
        cur.execute("DELETE FROM people WHERE id = ?", (person_id,))
        print(f"  people: {cur.rowcount} row(s)")
        db.commit()
        return 0
    finally:
        db.close()


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("name", nargs="?", help="person name to remove (case-insensitive)")
    parser.add_argument("--id", type=int, default=None, help="person id to remove")
    parser.add_argument(
        "--db", default=str(getattr(config, "DB_PATH", "assets/memory/people.db")),
        help="path to people.db",
    )
    args = parser.parse_args()
    if args.id is None and not args.name:
        parser.error("give a name or --id")
    return remove_person(args.db, person_id=args.id, name=args.name)


if __name__ == "__main__":
    raise SystemExit(main())
