"""Remove a phantom person minted from a misheard spoken name, and every trace.

Usage (dry run first — that is the default):
    venv/bin/python tools/cleanup_phantom_people.py
    venv/bin/python tools/cleanup_phantom_people.py --apply
    venv/bin/python tools/cleanup_phantom_people.py --id 9 --apply

Field 2026-08-27 13:38:44, unprompted into a seven-second lull: "I met someone
named Fuck once, which is honestly the most honest introduction this room has
ever offered." The row behind it was minted the night before at 2026-08-26
20:10:44, when the Jeopardy roster prompt heard "Jeremy, Bret, J T. Ah, fuck.
Never mind." and took the swear as a fourth contestant.
memory/name_validation.py rejects profanity at the source now; this cleans up the
rows that predate that.

Companion to tools/remove_phantom_person.py (the 2026-07-30 backchannel
incident). This one adds the three things that mattered here: it backs BOTH
databases up first, it prints every row before touching anything, and it refuses
to delete a person who has left any real trace (a face/voice print, a
conversation, a stored fact) unless --force says otherwise.
"""

from __future__ import annotations

import argparse
import sqlite3
import sys
from datetime import datetime, timezone
from pathlib import Path

_PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_PROJECT_ROOT))

import config  # noqa: E402

# Confirmed garbage: id 10, name "Fuck", minted 2026-08-26T20:10:44 from the
# Jeopardy roster prompt. id 9 "Jay" (same lane, 20:01:14) and id 5 "Jennifer"
# (Bret's sister, named in a relationship, never present) are NOT deleted by
# default — pass --id explicitly once a human has settled them.
_DEFAULT_TARGETS = ((10, "Fuck"),)

# Columns that hold a person reference. person_invited_topic is deliberately
# absent: it is a boolean flag on person_emotional_events, not a foreign key,
# and a substring match on "person" would have wiped real rows.
_PERSON_REF_COLUMNS = {
    "person_id", "from_person_id", "to_person_id",
    "person_a_id", "person_b_id", "subject_person_id",
}
# Attribution columns: the phantom is not the SUBJECT of these rows, so they are
# blanked rather than deleted — deleting would take a real person's data along.
_ATTRIBUTION_COLUMNS = {"described_by", "told_by"}

# A person with any of these is not a phantom. Refuse without --force.
_REAL_TRACE_TABLES = (
    "biometrics", "conversations", "person_facts", "person_events",
    "person_aliases", "voice_signatures", "conversation_log", "person_interests",
    "person_preferences", "person_qa",
)


def _resolve(path_value: str) -> Path:
    path = Path(path_value)
    return path if path.is_absolute() else (_PROJECT_ROOT / path)


def _backup(path: Path) -> Path:
    """Snapshot via the sqlite backup API — a plain file copy would miss the WAL."""
    stamp = datetime.now(timezone.utc).strftime("%Y%m%d-%H%M%S")
    dest = path.with_name(f"{path.name}.bak-{stamp}-phantom")
    src = sqlite3.connect(str(path))
    try:
        out = sqlite3.connect(str(dest))
        try:
            src.backup(out)
        finally:
            out.close()
    finally:
        src.close()
    return dest


def _tables(conn: sqlite3.Connection) -> list[str]:
    return [
        r[0] for r in conn.execute(
            "SELECT name FROM sqlite_master WHERE type='table' AND name NOT LIKE 'sqlite_%'"
        )
    ]


def _columns(conn: sqlite3.Connection, table: str) -> set[str]:
    return {c[1] for c in conn.execute(f"PRAGMA table_info({table})")}


def _real_trace_counts(conn: sqlite3.Connection, person_id: int) -> dict[str, int]:
    counts: dict[str, int] = {}
    present = set(_tables(conn))
    for table in _REAL_TRACE_TABLES:
        if table not in present or "person_id" not in _columns(conn, table):
            continue
        n = conn.execute(
            f"SELECT COUNT(*) FROM {table} WHERE person_id = ?", (person_id,)
        ).fetchone()[0]
        if n:
            counts[table] = int(n)
    return counts


def _plan_people(conn: sqlite3.Connection, person_id: int) -> list[tuple[str, str, int]]:
    plan: list[tuple[str, str, int]] = []
    for table in _tables(conn):
        for col in sorted(_columns(conn, table) & _PERSON_REF_COLUMNS):
            n = conn.execute(
                f"SELECT COUNT(*) FROM {table} WHERE {col} = ?", (person_id,)
            ).fetchone()[0]
            if n:
                plan.append((table, col, int(n)))
    return plan


def _plan_attribution(conn: sqlite3.Connection, person_id: int) -> list[tuple[str, str, int]]:
    plan: list[tuple[str, str, int]] = []
    for table in _tables(conn):
        for col in sorted(_columns(conn, table) & _ATTRIBUTION_COLUMNS):
            n = conn.execute(
                f"SELECT COUNT(*) FROM {table} WHERE {col} = ?", (person_id,)
            ).fetchone()[0]
            if n:
                plan.append((table, col, int(n)))
    return plan


def _rex_rows(conn: sqlite3.Connection, person_id: int, name: str) -> dict[str, list]:
    out: dict[str, list] = {}
    # The person_name clause catches the legacy NULL-person phantoms ("I met
    # Also.", rex.db id 25) that predate person_id being stamped on the episode.
    eps = conn.execute(
        "SELECT id, created_at, kind, summary FROM rex_episodes "
        "WHERE person_id = ? OR (person_id IS NULL AND person_name = ?)",
        (person_id, name),
    ).fetchall()
    if eps:
        out["rex_episodes"] = [tuple(r) for r in eps]
    bits = conn.execute(
        "SELECT id, topic, spoken_at FROM bit_ledger WHERE person_id = ?", (person_id,)
    ).fetchall()
    if bits:
        out["bit_ledger"] = [tuple(r) for r in bits]
    return out


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--id", type=int, action="append", default=None,
                        help="person id to remove (repeatable); defaults to the confirmed phantom")
    parser.add_argument("--apply", action="store_true",
                        help="actually delete; without it this only prints the plan")
    parser.add_argument("--force", action="store_true",
                        help="delete even when the person has real traces (prints/conversations)")
    parser.add_argument("--people-db",
                        default=str(getattr(config, "DB_PATH", "assets/memory/people.db")))
    parser.add_argument("--rex-db",
                        default=str(getattr(config, "REX_DB_PATH", "assets/memory/rex.db")))
    args = parser.parse_args()

    people_path = _resolve(args.people_db)
    rex_path = _resolve(args.rex_db)
    for path in (people_path, rex_path):
        if not path.exists():
            print(f"missing database: {path}")
            return 1

    people = sqlite3.connect(str(people_path))
    rex = sqlite3.connect(str(rex_path))
    try:
        targets: list[tuple[int, str]] = []
        if args.id:
            for pid in args.id:
                row = people.execute(
                    "SELECT id, name FROM people WHERE id = ?", (pid,)).fetchone()
                if row is None:
                    print(f"no person id={pid} — skipping")
                    continue
                targets.append((int(row[0]), str(row[1] or "")))
        else:
            for pid, expected in _DEFAULT_TARGETS:
                row = people.execute(
                    "SELECT id, name FROM people WHERE id = ?", (pid,)).fetchone()
                if row is None:
                    print(f"no person id={pid} — already gone")
                    continue
                actual = str(row[1] or "")
                # Ids are reused after a delete; never trust the number alone.
                if actual != expected:
                    print(f"REFUSING id={pid}: expected name {expected!r}, found {actual!r}")
                    continue
                targets.append((int(row[0]), actual))
        if not targets:
            print("nothing to do.")
            return 0

        blocked = False
        for pid, name in targets:
            print(f"\n=== person id={pid} name={name!r} ===")
            traces = _real_trace_counts(people, pid)
            if traces and not args.force:
                print(f"  REFUSING — this person has real traces: {traces}")
                print("  (re-run with --force only if a human has confirmed it is garbage)")
                blocked = True
                continue
            if traces:
                print(f"  --force: overriding real traces {traces}")
            for table, col, n in _plan_people(people, pid):
                print(f"  people.db  DELETE {table}.{col}: {n} row(s)")
            for table, col, n in _plan_attribution(people, pid):
                print(f"  people.db  NULL   {table}.{col}: {n} row(s)")
            print("  people.db  DELETE people.id: 1 row")
            for table, rows in _rex_rows(rex, pid, name).items():
                for row in rows:
                    print(f"  rex.db     DELETE {table}: {row}")

        targets = [t for t in targets if args.force or not _real_trace_counts(people, t[0])]
        if not targets:
            print("\nnothing left to remove.")
            return 1 if blocked else 0

        if not args.apply:
            print("\nDRY RUN — nothing was changed. Re-run with --apply.")
            return 0

        print(f"\nbacked up: {_backup(people_path)}")
        print(f"backed up: {_backup(rex_path)}")

        for pid, name in targets:
            for table, col, _n in _plan_people(people, pid):
                people.execute(f"DELETE FROM {table} WHERE {col} = ?", (pid,))
            for table, col, _n in _plan_attribution(people, pid):
                people.execute(f"UPDATE {table} SET {col} = NULL WHERE {col} = ?", (pid,))
            people.execute("DELETE FROM people WHERE id = ?", (pid,))
            rex.execute(
                "DELETE FROM rex_episodes WHERE person_id = ? OR "
                "(person_id IS NULL AND person_name = ?)",
                (pid, name),
            )
            rex.execute("DELETE FROM bit_ledger WHERE person_id = ?", (pid,))
        people.commit()
        rex.commit()
        print(f"\nremoved {len(targets)} person row(s) and every referencing row.")
        return 1 if blocked else 0
    finally:
        people.close()
        rex.close()


if __name__ == "__main__":
    raise SystemExit(main())
