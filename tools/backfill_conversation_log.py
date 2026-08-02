"""
tools/backfill_conversation_log.py — import logs/conversation-*.log history into
the conversation_log table.

Turns are written through live from add_to_transcript going forward; this seeds
the table with everything the on-disk conversation logs already hold, so dated
recall ("what did we talk about on July 12?") works for history predating the
feature. Idempotent: the table's UNIQUE(ts, speaker, text) makes re-runs no-ops.

Log line shapes (both carry a full local datetime):
    2026-08-01 22:38:29 | REX   | Boot successful. ...
    2026-08-01 22:39:19 | HEARD | Bret Benziger: I said I'm not going to ...

Usage:
    python tools/backfill_conversation_log.py            # dry run — counts only
    python tools/backfill_conversation_log.py --apply    # insert
"""

from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_ROOT))

from memory import database as db, people  # noqa: E402

_LINE_RE = re.compile(
    r"^(\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2}) \| (REX|HEARD)\s*\| (.*)$"
)
_HEARD_RE = re.compile(r"^([^:]{1,60}): (.*)$")

_person_cache: dict = {}


def _person_id(speaker: str):
    if speaker in _person_cache:
        return _person_cache[speaker]
    pid = None
    if not speaker.lower().startswith("unknown_voice"):
        try:
            row = people.find_person_by_name(speaker)
            pid = int(row["id"]) if row else None
        except Exception:
            pid = None
    _person_cache[speaker] = pid
    return pid


def backfill(apply: bool) -> tuple[int, int]:
    parsed = inserted = 0
    rows: list[tuple] = []
    for path in sorted((_ROOT / "logs").glob("conversation-*.log")):
        session_id = path.stem  # conversation-YYYY-MM-DD-HH-MM-SS
        for raw in path.read_text(errors="replace").splitlines():
            m = _LINE_RE.match(raw)
            if not m:
                continue
            ts, kind, body = m.group(1), m.group(2), m.group(3).strip()
            if not body:
                continue
            if kind == "REX":
                speaker, text = "Rex", body
            else:
                hm = _HEARD_RE.match(body)
                if not hm:
                    continue
                speaker, text = hm.group(1).strip(), hm.group(2).strip()
            if not text:
                continue
            parsed += 1
            if apply:
                rows.append((ts, ts[:10], session_id, speaker,
                             _person_id(speaker) if kind == "HEARD" else None, text))
    if apply and rows:
        n = db.executemany(
            """INSERT OR IGNORE INTO conversation_log
               (ts, day, session_id, speaker, person_id, text)
               VALUES (?, ?, ?, ?, ?, ?)""",
            rows,
        )
        inserted = int(n or 0)
    return parsed, inserted


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--apply", action="store_true",
                        help="insert rows (default: dry run, counts only)")
    args = parser.parse_args()
    parsed, inserted = backfill(apply=args.apply)
    if args.apply:
        print(f"parsed {parsed} turn(s); inserted {inserted} new row(s)")
    else:
        print(f"parsed {parsed} turn(s); re-run with --apply to insert")


if __name__ == "__main__":
    main()
