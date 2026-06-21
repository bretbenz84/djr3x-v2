#!/usr/bin/env python3
"""
tools/clean_memory.py — one-shot maintenance for the robot's memory DBs.

Pull the Phase A+B / Tier C+D changes, then run this ONCE on the robot to clean the
residue the automatic boot migration doesn't touch:

  people.db
    • run the dup-consolidation + orphan sweep (same as boot, forced so it runs even if
      user_version is already bumped),
    • delete junk-fragment interests ("him sassy") that predate the write-time junk gate,
    • cap runaway evidence_count (the old "13 confirmations" on within-session chatter).

  rex.db (Rex's diary)
    • collapse EXACT duplicate episodes ("I met Bret" x6 → 1, "I saw Bret" x7 → 1),
    • null the person link on episodes that point at a deleted person id OR carry a name
      that no longer matches that id (recycled-id mislabels — "Sarah" episodes on Bret's id).

SAFE BY DEFAULT: this is a DRY RUN unless you pass --apply. With --apply it makes a
timestamped backup of each DB first (unless --no-backup). It never deletes a PERSON; if
you want to drop a test/dummy person, pass --drop-person ID (uses the normal cascading
delete, which also cleans that person's diary entries).

Usage:
    venv/bin/python tools/clean_memory.py              # dry run — shows what it would do
    venv/bin/python tools/clean_memory.py --apply       # do it (with backups)
    venv/bin/python tools/clean_memory.py --apply --drop-person 2
"""

from __future__ import annotations

import argparse
import sqlite3
import sys
import time
from pathlib import Path

_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

import config  # noqa: E402
from memory import database as db, dedup, rex_db  # noqa: E402


def _people_path() -> Path:
    return Path(db._DB_FILE)


def _rex_path() -> Path:
    return Path(rex_db.db_path())


def _backup(path: Path) -> Path | None:
    """Consistent snapshot via the SQLite backup API (WAL-safe)."""
    if not path.exists():
        return None
    stamp = time.strftime("%Y%m%d-%H%M%S")
    dest = path.with_name(f"{path.name}.bak-clean-{stamp}")
    src = sqlite3.connect(str(path))
    try:
        out = sqlite3.connect(str(dest))
        with out:
            src.backup(out)
        out.close()
    finally:
        src.close()
    return dest


def _first_token(name: str) -> str:
    return (name or "").strip().lower().split()[0] if (name or "").strip() else ""


def _names_compatible(a: str, b: str) -> bool:
    """A diary name snapshot is compatible with the current person name if one contains
    the other ("Bret" vs "Bret Benziger") or they share a first token. Otherwise the
    episode predates an id recycle/rename and its person link is stale."""
    a, b = (a or "").strip().lower(), (b or "").strip().lower()
    if not a or not b:
        return True  # nothing to contradict
    if a in b or b in a:
        return True
    return _first_token(a) == _first_token(b)


# ── people.db ─────────────────────────────────────────────────────────────────

def clean_people(apply: bool, cap_evidence: int) -> dict:
    report: dict = {}

    # 1. Structural: migrations (orphan sweep) + a forced dup-consolidation pass.
    if apply:
        try:
            db.verify_schema()  # runs migrations + orphan sweep + the one-time dedup
        except Exception as exc:
            print(f"  ! verify_schema warning: {exc}")
        report["consolidated"] = dedup.consolidate_all()
    else:
        # Dry run: count what consolidation WOULD collapse without writing.
        pid_rows = db.fetchall("SELECT id FROM people")
        would = 0
        for r in pid_rows:
            ints = [dict(x) for x in db.fetchall(
                "SELECT name FROM person_interests WHERE person_id=? ORDER BY id", (r["id"],))]
            for cluster in dedup._cluster(ints, "name"):
                would += max(0, len(cluster) - 1)
        report["consolidated"] = {"interests_removed_est": would}

    # 2. Junk-fragment interests.
    junk = [
        dict(r) for r in db.fetchall("SELECT id, name FROM person_interests")
        if dedup.looks_like_junk_interest(r["name"])
    ]
    report["junk_interests"] = [(r["id"], r["name"]) for r in junk]
    if apply and junk:
        ids = [int(r["id"]) for r in junk]
        ph = ",".join("?" for _ in ids)
        db.execute(f"DELETE FROM person_interests WHERE id IN ({ph})", tuple(ids))

    # 3. Cap inflated evidence_count.
    if cap_evidence > 0:
        inflated = db.fetchall(
            "SELECT COUNT(*) AS n FROM person_facts WHERE evidence_count > ?", (cap_evidence,))
        report["evidence_capped"] = int(inflated[0]["n"]) if inflated else 0
        if apply and report["evidence_capped"]:
            db.execute(
                "UPDATE person_facts SET evidence_count=? WHERE evidence_count > ?",
                (cap_evidence, cap_evidence))
    else:
        report["evidence_capped"] = 0

    # 4. Flag suspicious people (never auto-deleted).
    report["suspicious_people"] = [
        (r["id"], r["name"]) for r in db.fetchall("SELECT id, name FROM people")
        if any(t in (r["name"] or "").lower() for t in ("test", "dummy", "delete me", "example"))
    ]
    return report


def drop_people(ids: list[int], apply: bool) -> list[int]:
    from memory import people
    dropped = []
    for pid in ids:
        row = people.get_person(pid)
        if not row:
            print(f"  ! person id={pid} not found — skipping")
            continue
        print(f"  {'DROP' if apply else 'would drop'} person id={pid} ({row.get('name')})"
              " + all their memory + diary entries")
        if apply:
            people.delete_person(pid)
        dropped.append(pid)
    return dropped


# ── rex.db ────────────────────────────────────────────────────────────────────

def clean_rex(apply: bool) -> dict:
    report: dict = {"available": False}
    if not _rex_path().exists():
        return report
    report["available"] = True

    before = rex_db.fetchone("SELECT COUNT(*) AS n FROM rex_episodes")
    n_before = int(before["n"]) if before else 0

    # 1. Exact-duplicate episodes → keep the earliest (MIN id) of each group.
    dup_groups = rex_db.fetchall(
        "SELECT kind, person_id, COALESCE(summary,'') s, COUNT(*) n "
        "FROM rex_episodes GROUP BY kind, person_id, s HAVING n > 1")
    report["dup_groups"] = len(dup_groups)
    report["dup_rows_removed"] = sum(int(g["n"]) - 1 for g in dup_groups)
    if apply and dup_groups:
        rex_db.execute(
            "DELETE FROM rex_episodes WHERE id NOT IN ("
            "  SELECT MIN(id) FROM rex_episodes GROUP BY kind, person_id, COALESCE(summary,''))")

    # 2. Stale person links: id no longer exists, OR name snapshot no longer matches.
    people_map = {int(r["id"]): (r["name"] or "") for r in db.fetchall("SELECT id, name FROM people")}
    stale = []
    for r in rex_db.fetchall(
            "SELECT id, person_id, person_name FROM rex_episodes WHERE person_id IS NOT NULL"):
        pid = int(r["person_id"])
        if pid not in people_map:
            stale.append(int(r["id"]))                                   # orphan id
        elif not _names_compatible(r["person_name"], people_map[pid]):
            stale.append(int(r["id"]))                                   # recycled/renamed
    report["stale_links"] = len(stale)
    if apply and stale:
        ph = ",".join("?" for _ in stale)
        rex_db.execute(
            f"UPDATE rex_episodes SET person_id=NULL, person_name=NULL WHERE id IN ({ph})",
            tuple(stale))

    after = rex_db.fetchone("SELECT COUNT(*) AS n FROM rex_episodes")
    report["rows_before"] = n_before
    report["rows_after"] = int(after["n"]) if after else n_before
    return report


# ── main ──────────────────────────────────────────────────────────────────────

def main() -> int:
    ap = argparse.ArgumentParser(description="Clean the robot's memory databases.")
    ap.add_argument("--apply", action="store_true", help="Write changes (default: dry run).")
    ap.add_argument("--no-backup", action="store_true", help="Skip the pre-change DB backup.")
    ap.add_argument("--cap-evidence", type=int, default=3,
                    help="Cap evidence_count at N (0 = leave it). Default 3.")
    ap.add_argument("--drop-person", type=int, action="append", default=[],
                    help="Delete a person id entirely (cascades to their diary). Repeatable.")
    args = ap.parse_args()

    mode = "APPLY" if args.apply else "DRY RUN (pass --apply to write)"
    print(f"=== clean_memory — {mode} ===")
    print(f"people.db: {_people_path()}")
    print(f"rex.db:    {_rex_path()}  ({'present' if _rex_path().exists() else 'absent'})")

    if args.apply and not args.no_backup:
        for p in (_people_path(), _rex_path()):
            dest = _backup(p)
            if dest:
                print(f"  backed up {p.name} → {dest.name}")

    print("\n[people.db]")
    pr = clean_people(args.apply, args.cap_evidence)
    print(f"  consolidation: {pr['consolidated']}")
    print(f"  junk-fragment interests {'removed' if args.apply else 'to remove'}: "
          f"{pr['junk_interests'] or 'none'}")
    print(f"  facts with evidence_count capped at {args.cap_evidence}: {pr['evidence_capped']}")
    if pr["suspicious_people"]:
        print(f"  ⚠ suspicious people (NOT auto-deleted — use --drop-person ID): "
              f"{pr['suspicious_people']}")
    if args.drop_person:
        print("  drop-person:")
        drop_people(args.drop_person, args.apply)

    print("\n[rex.db]")
    rr = clean_rex(args.apply)
    if not rr["available"]:
        print("  (no rex.db yet — nothing to clean)")
    else:
        print(f"  duplicate episode groups: {rr['dup_groups']} "
              f"({rr['dup_rows_removed']} rows {'removed' if args.apply else 'to remove'})")
        print(f"  stale person links {'nulled' if args.apply else 'to null'}: {rr['stale_links']}")
        print(f"  episode rows: {rr['rows_before']} → {rr['rows_after']}")

    if not args.apply:
        print("\nDry run only — re-run with --apply to make these changes.")
    else:
        print("\nDone.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
