"""
awareness/interoception.py — System health awareness for DJ-R3X.
"""

import logging
import re
import subprocess
import sys
import threading
import time
from pathlib import Path
from typing import Optional

_PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

import config
from world_state import world_state

try:
    import psutil as _psutil
    _PSUTIL_OK = True
except ImportError:
    _PSUTIL_OK = False

_log = logging.getLogger(__name__)

# ─────────────────────────────────────────────────────────────────────────────
# Module-level state — all in-memory, resets on process restart
# ─────────────────────────────────────────────────────────────────────────────

_process_start = time.monotonic()

_state_lock = threading.Lock()
_session_interaction_count: int = 0
_last_interaction_at: Optional[float] = None  # time.monotonic() timestamp

# cpu_temp is read via powermetrics which takes ~1 s — cache it in the
# background loop so get_system_state() never blocks a caller.
_cached_cpu_temp: Optional[float] = None

_stop_event = threading.Event()
_thread: Optional[threading.Thread] = None


# ─────────────────────────────────────────────────────────────────────────────
# Internal helpers
# ─────────────────────────────────────────────────────────────────────────────

def _read_cpu_temp_macos() -> Optional[float]:
    """
    Read CPU die temperature via powermetrics. Requires passwordless sudo (-n flag).
    Returns None on any failure rather than blocking or raising.
    """
    try:
        result = subprocess.run(
            ["sudo", "-n", "powermetrics",
             "--samplers", "smc", "-i", "1000", "-n", "1"],
            capture_output=True, text=True, timeout=6,
        )
        for line in result.stdout.splitlines():
            if "CPU die temperature" in line:
                m = re.search(r"([\d.]+)\s*C", line)
                if m:
                    return float(m.group(1))
    except Exception:
        pass
    return None


# ─────────────────────────────────────────────────────────────────────────────
# Public API
# ─────────────────────────────────────────────────────────────────────────────

def get_system_state() -> dict:
    """
    Return current system state dict with keys:
    uptime_seconds, cpu_temp, cpu_load, session_interaction_count, last_interaction_ago.
    cpu_temp is read from the background-refreshed cache (never blocks).
    cpu_load is a 0.0–1.0 fraction from psutil, or None if unavailable.
    """
    with _state_lock:
        count = _session_interaction_count
        last_at = _last_interaction_at
        cpu_temp = _cached_cpu_temp

    uptime = int(time.monotonic() - _process_start)
    cpu_load = (_psutil.cpu_percent(interval=None) / 100.0) if _PSUTIL_OK else None
    last_ago = round(time.monotonic() - last_at) if last_at is not None else None

    return {
        "uptime_seconds": uptime,
        "cpu_temp": cpu_temp,
        "cpu_load": cpu_load,
        "session_interaction_count": count,
        "last_interaction_ago": last_ago,
    }


def increment_interaction_count() -> None:
    """Increment the session interaction counter without updating the timestamp."""
    global _session_interaction_count
    with _state_lock:
        _session_interaction_count += 1


def record_interaction() -> None:
    """Mark a completed interaction — increments the count and updates the timestamp."""
    global _session_interaction_count, _last_interaction_at
    with _state_lock:
        _session_interaction_count += 1
        _last_interaction_at = time.monotonic()


def start_periodic_update(interval: Optional[float] = None) -> None:
    """
    Start a daemon thread that writes get_system_state() into world_state.self_state
    and refreshes the cached CPU temperature in the background.
    """
    global _thread, _stop_event

    if _thread and _thread.is_alive():
        return

    interval = interval or getattr(config, "INTEROCEPTION_UPDATE_INTERVAL_SECS", 5.0)
    _stop_event.clear()

    def _loop() -> None:
        global _cached_cpu_temp
        while not _stop_event.is_set():
            try:
                temp = _read_cpu_temp_macos()
                with _state_lock:
                    _cached_cpu_temp = temp

                state = get_system_state()
                self_state = world_state.get("self_state")
                self_state.update(state)
                world_state.update("self_state", self_state)
            except Exception as exc:
                _log.error("interoception update failed: %s", exc)
            _stop_event.wait(interval)

    _thread = threading.Thread(target=_loop, daemon=True, name="interoception")
    _thread.start()
    _log.info("interoception started (interval=%.1fs)", interval)


def stop() -> None:
    """Stop the background update thread."""
    global _thread
    _stop_event.set()
    if _thread:
        _thread.join(timeout=5)
        _thread = None
    _log.info("interoception stopped")
