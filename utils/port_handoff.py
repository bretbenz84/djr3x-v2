"""
Serial-port handoff — wait for the menu bar companions to let go before we open.

Three always-on menu bar LaunchAgents hold the robot's serial ports while Rex is
off: tools/rex_battery_menubar.py (motion ESP32), tools/rex_servo_menubar.py
(Maestro) and tools/rex_led_menubar.py (head + chest Arduinos). Each of them
polls the single-instance flock about once a second and closes its port the
moment main.py takes it.

That flock — not the supervisor — is the entire handoff. rex_supervisor.py never
touches a serial port; it only spawns main.py. So a manual
`venv/bin/python main.py` already triggers the same release; it just used to
arrive too early. main.py took the lock and opened the ports inside the same
second, ahead of the companions' 1 Hz poll. servos.py and motion.py survived on
their 3x1s connect retries; leds_head/leds_chest opened exactly once, so a
manual launch while the LED console held the boards lost the LEDs for the whole
session. This module is the missing pause.

**Detection is `lsof`, deliberately.** Probing by opening the device would defeat
the purpose: an open toggles DTR and reboots both Arduinos, so every startup
would pay an extra board reset — and pyserial's default open takes no lock at
all (it only flocks when `exclusive=True`), so a probe would "succeed" against a
port a companion is still using and tell us nothing. lsof never touches the
device, and it sees *any* holder — an Arduino IDE serial monitor or a stray
tools/ script counts too, which is exactly what the owner wants when the ask is
"let me start Rex without unloading things first".

Failing open is the rule throughout: if lsof is missing or unusable we sleep past
one companion poll and continue. A handoff helper must never be the reason the
robot won't start.
"""

from __future__ import annotations

import logging
import os
import shutil
import subprocess
import time
from typing import Callable, Optional, Sequence

_log = logging.getLogger(__name__)

# Ships with macOS; PATH lookup covers unusual environments.
_LSOF = "/usr/sbin/lsof"

# One lsof call must never wedge startup.
_LSOF_TIMEOUT_SECS = 5.0

# Long enough to cover the companions' ~1 Hz lock poll plus their close, short
# enough that a genuinely stuck port doesn't noticeably delay boot.
DEFAULT_TIMEOUT_SECS = 5.0
DEFAULT_POLL_SECS = 0.15

# Used only when lsof can't be run: sleep past one companion poll rather than
# skipping the handoff entirely.
DEFAULT_BLIND_GRACE_SECS = 1.5


def _lsof_path() -> Optional[str]:
    if os.path.exists(_LSOF):
        return _LSOF
    return shutil.which("lsof")


def holders(devices: Sequence[str]) -> "dict[str, list[int]] | None":
    """Map each device path to the pids holding it open, excluding this process.

    Returns `{}` when lsof ran and found nothing, and `None` when lsof could not
    be run or failed — "free" and "can't tell" are different answers and the
    caller must not confuse them.
    """
    wanted = [d for d in devices if d]
    if not wanted:
        return {}

    lsof = _lsof_path()
    if lsof is None:
        return None
    try:
        proc = subprocess.run(
            [lsof, "-Fpn", "--", *wanted],
            capture_output=True,
            text=True,
            timeout=_LSOF_TIMEOUT_SECS,
        )
    except (OSError, subprocess.SubprocessError):
        return None
    # Exit status 1 means "at least one of these files has no open holder" —
    # the normal case here, NOT an error. Only a stranger status is a failure.
    if proc.returncode not in (0, 1):
        return None

    by_basename = {os.path.basename(d): d for d in wanted}
    exact = set(wanted)
    me = os.getpid()

    found: dict[str, list[int]] = {}
    pid: Optional[int] = None
    for line in proc.stdout.splitlines():
        if not line:
            continue
        tag, value = line[0], line[1:]
        if tag == "p":
            try:
                pid = int(value)
            except ValueError:
                pid = None
        elif tag == "n" and pid is not None and pid != me:
            device = value if value in exact else by_basename.get(os.path.basename(value))
            if device is None:
                continue
            pids = found.setdefault(device, [])
            if pid not in pids:
                pids.append(pid)
    return found


def describe_holder(pid: int) -> str:
    """`rex_servo_menubar.py (pid 750)` — best effort, for log lines only."""
    try:
        proc = subprocess.run(
            ["ps", "-p", str(pid), "-o", "command="],
            capture_output=True,
            text=True,
            timeout=2.0,
        )
        command = proc.stdout.strip()
    except (OSError, subprocess.SubprocessError):
        command = ""
    script = next(
        (os.path.basename(part) for part in reversed(command.split()) if part.endswith(".py")),
        "",
    )
    name = script or (os.path.basename(command.split()[0]) if command else "unknown")
    return f"{name} (pid {pid})"


def wait_for_release(
    ports: Sequence[tuple[str, str]],
    *,
    timeout: float = DEFAULT_TIMEOUT_SECS,
    poll: float = DEFAULT_POLL_SECS,
    blind_grace: float = DEFAULT_BLIND_GRACE_SECS,
    log: Optional[logging.Logger] = None,
    sleep: Callable[[float], None] = time.sleep,
    monotonic: Callable[[], float] = time.monotonic,
) -> "list[str]":
    """Block until nothing else holds `ports`, or `timeout` elapses.

    `ports` is a sequence of `(label, device)` pairs; entries with no device are
    skipped. Returns the labels still held when we gave up — empty on success,
    which is also the instant answer when the ports were free to begin with.
    """
    wanted = [(label, device) for label, device in ports if device]
    if not wanted or timeout <= 0:
        return []  # timeout=0 is the opt-out: no lsof, no wait, no log

    log = log or _log
    devices = [device for _, device in wanted]

    held = holders(devices)
    if held is None:
        log.info(
            "Serial handoff: lsof unavailable — pausing %.1fs for the menu bar "
            "companions to release the ports.",
            blind_grace,
        )
        sleep(blind_grace)
        return []
    if not held:
        return []  # fast path: nobody is holding anything, say nothing

    log.info(
        "Serial handoff: waiting for %s.",
        "; ".join(
            f"{label} on {device} held by "
            + ", ".join(describe_holder(pid) for pid in held[device])
            for label, device in wanted
            if device in held
        ),
    )

    deadline = monotonic() + max(0.0, timeout)
    started = monotonic()
    while True:
        if monotonic() >= deadline:
            break
        sleep(poll)
        held = holders(devices)
        if held is None:
            held = {}  # lsof went away mid-wait; we already paused, so proceed
        if not held:
            log.info("Serial handoff: all ports released after %.1fs.", monotonic() - started)
            return []

    stuck = [label for label, device in wanted if device in held]
    log.warning(
        "Serial handoff: %s still held after %.1fs (%s). Connecting anyway — "
        "quit the menu bar app holding it if that hardware comes up disabled.",
        ", ".join(stuck),
        timeout,
        "; ".join(
            describe_holder(pid) for device in held for pid in held[device]
        ),
    )
    return stuck
