"""
intelligence/action_result.py — one correlated record per body action.

Lean Brain restructuring, phase 4. A motion command used to leave its truth
scattered: `motion.send()` returned a seq (or None), the firmware's done frame
went to a reader-thread callback, the swing check could SHRINK a turn on the way
out, the compass verification ran later on another thread, and a refusal was a
three-second global. None of it said, in one place, "you asked for X, we sent Y,
it ended Z, and the compass measured W". This record does. It is created when a
command is issued or refused, updated by the done frame and the compass check,
and rendered for the reply model by intelligence/conversation_state.py.

Rules: `None` from an executor, or a thread having started, is never a success.
`requested_deg` is what the human/plan asked for; `attempted_deg` is what was
actually sent after the swing check or a heading alternative; `measured_deg` is
the compass-verified rotation when available. Status vocabulary is the
firmware's plus the host's own words.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, asdict, field
from typing import Optional

# Host + firmware status vocabulary. Firmware done results: completed / blocked /
# aborted / superseded; host: running / refused / timeout / cancelled /
# not_settled / error / suppressed.
STATUSES = frozenset({
    "running", "completed", "partial", "blocked", "aborted", "superseded",
    "refused", "timeout", "cancelled", "not_settled", "error", "suppressed", "unknown",
})


@dataclass
class ActionResult:
    verb: str                                  # turn / move / come here / arc / route step
    detail: str = ""                           # human words: "left 90°", "forward 0.30 m"
    seq: Optional[int] = None                  # firmware sequence id, None when never sent
    status: str = "running"
    reason: str = ""                           # refusal / failure code from the host or firmware
    requested_deg: Optional[float] = None      # what was asked
    attempted_deg: Optional[float] = None      # what was sent (after shrink / alternative)
    measured_deg: Optional[float] = None       # compass-verified rotation, when checked
    alternative: str = ""                      # e.g. "asked left 180°, went right 180° (swing)"
    at: float = field(default_factory=time.monotonic)
    ended_at: Optional[float] = None

    def finish(self, status: str, *, reason: str = "") -> None:
        status = str(status or "unknown").strip().lower()
        self.status = status if status in STATUSES else status or "unknown"
        if reason:
            self.reason = str(reason)
        self.ended_at = time.monotonic()

    @property
    def shrunk(self) -> bool:
        return (self.requested_deg is not None and self.attempted_deg is not None
                and abs(abs(self.requested_deg) - abs(self.attempted_deg)) > 1.0)

    @property
    def ok(self) -> bool:
        return self.status == "completed"

    def as_dict(self) -> dict:
        d = asdict(self)
        d["shrunk"] = self.shrunk
        return d
