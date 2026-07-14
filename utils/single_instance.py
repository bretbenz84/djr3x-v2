"""
Single-instance lock for the DJ-R3X controller (main.py).

Why this exists: a tiny always-on supervisor (rex_supervisor.py, started by a
macOS LaunchAgent) listens only for the "wake up rex" wake word and launches the
full controller. The supervisor must never spawn a SECOND main.py while one is
already running — including while the running one is just asleep. The shared
truth is this lock file: main.py holds it for its whole lifetime, the supervisor
checks it before launching and stays dormant while it is held.

Mechanics: an advisory `flock` (LOCK_EX | LOCK_NB) on a file under a runtime
directory. flock is auto-released by the OS when the holding process exits —
including a crash or `kill -9` — so a dead controller never strands the lock and
the supervisor resumes listening on its own. The file's bytes hold the owner pid
purely for humans/logs; the lock itself is the source of truth, not the pid text.

Cross-process API:
    acquire() -> bool        # main.py: True if we now hold it, False if busy
    is_held_by_other() -> bool  # supervisor: True if some OTHER process holds it
    read_owner_pid() -> int|None
    release()                # optional; OS also releases on exit

The lock path is `DJR3X_LOCK_PATH` if set, else `<tmpdir>/djr3x-main.lock`.
"""

from __future__ import annotations

import errno
import fcntl
import os
import tempfile
import time
from pathlib import Path
from typing import Optional


def lock_path() -> Path:
    """Resolve the lock file path (env override, else a stable tmpdir location)."""
    override = os.environ.get("DJR3X_LOCK_PATH", "").strip()
    if override:
        return Path(override).expanduser()
    return Path(tempfile.gettempdir()) / "djr3x-main.lock"


# Module-held file handle. Kept open for the process lifetime so the flock
# persists; closing it (or exiting) drops the lock.
_handle = None  # type: ignore[var-annotated]


def acquire() -> bool:
    """Try to take the single-instance lock for THIS process.

    Returns True if we now hold it, False if another live process holds it.
    Idempotent: calling again while already held returns True.
    """
    global _handle
    if _handle is not None:
        return True

    path = lock_path()
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
    except OSError:
        pass

    # Open (not truncate) so a busy holder's pid bytes survive our failed attempt.
    handle = open(path, "a+")
    # A few brief retries: is_held_by_other() probers hold a momentary SHARED
    # lock, which would fail our exclusive attempt if we land inside that
    # microseconds-wide window. A real owner still holds EX across all retries.
    acquired = False
    for i in range(5):
        try:
            fcntl.flock(handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
            acquired = True
            break
        except OSError as exc:
            if exc.errno not in (errno.EACCES, errno.EAGAIN):
                handle.close()
                raise
            if i < 4:
                time.sleep(0.05)
    if not acquired:
        handle.close()
        return False  # someone else holds it

    # We hold it — (re)write our pid as the human-readable owner record.
    try:
        handle.seek(0)
        handle.truncate(0)
        handle.write(f"{os.getpid()}\n")
        handle.flush()
        os.fsync(handle.fileno())
    except OSError:
        pass

    _handle = handle
    return True


def is_held() -> bool:
    """True if THIS process currently holds the lock."""
    return _handle is not None


def is_held_by_other() -> bool:
    """True if some OTHER live process holds the lock.

    Used by the supervisor AND the menu bar utilities to decide whether a
    controller is already running (awake or asleep). Probes with a SHARED
    non-blocking lock: main.py's exclusive lock blocks it (held → True), while
    any number of concurrent probers hold SH together without blocking each
    other. The probe MUST NOT be LOCK_EX — with several 1 Hz pollers (the
    supervisor plus each menu bar app), exclusive probes collided with each
    other and randomly reported "Rex is running" to a peer, which then flapped
    its serial port closed/open (field bug 2026-07-13: each flap rebooted the
    ESP32 and dropped the gamepad).
    """
    if _handle is not None:
        return False  # we hold it; not "other"

    path = lock_path()
    if not path.exists():
        return False

    try:
        probe = open(path, "a+")
    except OSError:
        # Can't open — assume not held rather than block the supervisor forever.
        return False
    try:
        fcntl.flock(probe.fileno(), fcntl.LOCK_SH | fcntl.LOCK_NB)
    except OSError as exc:
        if exc.errno in (errno.EACCES, errno.EAGAIN):
            return True  # a live process holds it
        return False
    else:
        # We got it → nobody held it. Release immediately; we were only probing.
        try:
            fcntl.flock(probe.fileno(), fcntl.LOCK_UN)
        except OSError:
            pass
        return False
    finally:
        probe.close()


def read_owner_pid() -> Optional[int]:
    """Best-effort read of the pid recorded in the lock file (for logs/UX)."""
    path = lock_path()
    try:
        text = path.read_text().strip()
    except OSError:
        return None
    if not text:
        return None
    try:
        return int(text.splitlines()[0])
    except (ValueError, IndexError):
        return None


def release() -> None:
    """Release the lock if held. The OS also does this automatically on exit."""
    global _handle
    if _handle is None:
        return
    try:
        fcntl.flock(_handle.fileno(), fcntl.LOCK_UN)
    except OSError:
        pass
    try:
        _handle.close()
    except OSError:
        pass
    _handle = None
