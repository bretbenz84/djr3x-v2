"""
One-shot cleanup for the two rows the 2026-08-29 11:15 intro misread wrote.

The code fix stops it happening again; it does not reach back into people.db.
Run this once to undo what that session stored:

  venv/bin/python tools/cleanup_2026_08_29_intro_misread.py          # dry run
  venv/bin/python tools/cleanup_2026_08_29_intro_misread.py --apply

  * biometrics id 56 — Bret's voice ("I didn't leave. I just turned around.")
    enrolled onto PJ Thomas (person 7) at 18:21:46. It is in PJ's centroid, so
    every later voice match is scored against a print set that contains someone
    else. This is the one worth removing.
  * person_facts 132 / 133 — "Bret Benziger and PJ: PJ is not here. This is
    Bret." filed as their connection story on both people at confidence 0.90.

Each target is verified before deletion; anything that no longer matches is
skipped rather than guessed at.
"""

from __future__ import annotations

import sqlite3
import sys
from pathlib import Path

DB = Path(__file__).resolve().parent.parent / "assets" / "memory" / "people.db"

BAD_BIOMETRIC = (56, 7, "voice", "2026-08-29T18:21:46")
BAD_FACT_VALUE = "Bret Benziger and PJ: PJ is not here. This is Bret."


def main() -> int:
    apply = "--apply" in sys.argv
    conn = sqlite3.connect(DB)
    conn.row_factory = sqlite3.Row
    planned: list[tuple[str, tuple]] = []

    bio_id, pid, btype, stamp_prefix = BAD_BIOMETRIC
    row = conn.execute(
        "SELECT id, person_id, type, created_at FROM biometrics WHERE id = ?",
        (bio_id,),
    ).fetchone()
    if (
        row
        and row["person_id"] == pid
        and row["type"] == btype
        and str(row["created_at"]).startswith(stamp_prefix)
    ):
        print(f"  biometrics id={bio_id} person_id={pid} {btype} {row['created_at']}")
        planned.append(("DELETE FROM biometrics WHERE id = ?", (bio_id,)))
    else:
        print(f"  biometrics id={bio_id}: no longer matches — skipping")

    for row in conn.execute(
        "SELECT id, person_id, key, value FROM person_facts WHERE value = ?",
        (BAD_FACT_VALUE,),
    ):
        print(f"  person_facts id={row['id']} person_id={row['person_id']} "
              f"{row['key']}: {row['value']}")
        planned.append(("DELETE FROM person_facts WHERE id = ?", (row["id"],)))

    if not planned:
        print("Nothing to clean up.")
        return 0
    if not apply:
        print(f"\nDry run — {len(planned)} row(s) would be deleted. "
              f"Re-run with --apply.")
        return 0

    for sql, params in planned:
        conn.execute(sql, params)
    conn.commit()
    print(f"\nDeleted {len(planned)} row(s).")

    remaining = conn.execute(
        "SELECT COUNT(*) AS n FROM biometrics WHERE person_id = ? AND type = 'voice'",
        (pid,),
    ).fetchone()["n"]
    print(f"PJ Thomas (person {pid}) now has {remaining} voice print(s).")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
