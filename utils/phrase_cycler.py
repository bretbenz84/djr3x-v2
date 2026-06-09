"""Pick a phrase from a list, cycling so launches don't repeat consecutively.

Recently-used phrases are tracked in a small untracked JSON state file (under the
gitignored assets/state/) so consecutive runs differ; when the whole set has been
used the cycle restarts WITHOUT an immediate back-to-back repeat of the last phrase.
Shared by the startup boot/ready filler lines and the shutdown sign-off lines.
"""

import hashlib
import json
import logging
from pathlib import Path

# random is used for the pick; isolated so callers don't have to seed anything.
import random

_log = logging.getLogger(__name__)


def line_key(line: str) -> str:
    """Stable per-line key so the state file survives edits/reordering of the list."""
    return hashlib.sha1(line.encode("utf-8")).hexdigest()[:12]


def select_cycling_line(lines, state_path) -> str:
    """Return a line from ``lines``, cycling without back-to-back repeats across runs.

    ``state_path`` is the JSON file (created/updated) that remembers which lines have
    been used this cycle and which was last. Returns "" when ``lines`` is empty, and
    the sole line (no state write) when there is exactly one. Never raises — state
    read/write failures degrade to a fresh pick.
    """
    lines = [str(item).strip() for item in (lines or []) if str(item).strip()]
    if not lines:
        return ""
    if len(lines) == 1:
        return lines[0]

    state_path = Path(state_path)
    used: list[str] = []
    last: str = ""
    try:
        if state_path.exists():
            data = json.loads(state_path.read_text(encoding="utf-8"))
            if isinstance(data, dict):
                used = [str(k) for k in data.get("used", []) if isinstance(k, str)]
                last = str(data.get("last", "") or "")
    except Exception as exc:
        _log.debug("Could not read phrase-cycle state %s (%s); starting fresh.", state_path, exc)

    keys = {line_key(line): line for line in lines}
    # Drop any stale keys (lines that were edited/removed) from the used set.
    used = [k for k in used if k in keys]

    candidates = [line for key, line in keys.items() if key not in used]
    if not candidates:
        # Whole cycle exhausted — restart, but don't immediately repeat the last line.
        used = []
        candidates = [line for key, line in keys.items() if key != last] or lines

    chosen = random.choice(candidates)
    chosen_key = line_key(chosen)
    used.append(chosen_key)

    try:
        state_path.parent.mkdir(parents=True, exist_ok=True)
        state_path.write_text(
            json.dumps({"used": used, "last": chosen_key}, indent=2),
            encoding="utf-8",
        )
    except Exception as exc:
        _log.debug("Could not persist phrase-cycle state %s: %s", state_path, exc)

    return chosen
