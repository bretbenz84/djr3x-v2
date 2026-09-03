"""
reSpeaker Flex XVF3800 direction-of-arrival (DoA) poller — a bearing for every
voice, in the base frame.

The XVF3800 tracks where speech comes from on-chip and publishes it as
DOA_VALUE (0-359°, plus a speech-detected flag) on the same USB control
endpoint tools/flex_ctl.py reads. This module polls that register a few times a
second on a daemon thread, keeps a short history, and answers one question for
the rest of the system: "over this time window, where did the talker's voice
come from?" (`bearing_between`). Consumers: the come-here search
(intelligence/motion_agency.py) layers it with the radar ring, and the
off-camera speaker gaze search (intelligence/consciousness.py) aims the first
glance with it. Both treat it as a HINT — the camera still decides who is there.

Frame and convention (measured 2026-09-02 with tools/flex_doa.py, the ring
mounted with its printed 0° edge forward): chip 0° = dead ahead, 90° = Rex's
LEFT, 270° = his RIGHT, 180° = behind. The base frame is + = left/CCW (the
turn command convention), so base bearing = wrap180(sign * (doa - offset)) with
sign +1 and offset 0 — `FLEX_DOA_SIGN` / `FLEX_DOA_FORWARD_OFFSET_DEG` exist
for a re-mount. `FLEX_DOA_MOUNT="head"` subtracts the neck yaw sampled at each
poll (the ring turns with the head; the base frame does not).

Why a dominant CLUSTER rather than a median: between a talker's words the
register falls back to whatever else the room offers — in the 2026-09-02
"right" run it snapped to ~86° (a steady source on the robot's left) on a third
of the samples, dragging the plain median from 270° to 291°. The largest group
of mutually-agreeing samples over the window is the talker; the fallback
readings are the minority and are ignored.

Read-only: nothing here writes to the chip. All operations are no-ops when the
poller is disabled (FLEX_DOA_ENABLED off) or the device is absent.
"""

import logging
import math
import threading
import time
from collections import deque
from typing import Callable, Optional

import config

_log = logging.getLogger(__name__)

_lock = threading.Lock()
_samples: deque = deque()          # (t_mono, doa_raw_deg, base_bearing_deg, speech, energy, moving)
_last_moving_at: float = 0.0       # last poll that saw the base in motion (settle window anchor)
_thread: Optional[threading.Thread] = None
_stop = threading.Event()
_dev = None
_dev_lock = threading.Lock()
_status: dict = {"enabled": False, "connected": False, "reads": 0, "errors": 0, "last_error": ""}

# Optional: the head's yaw off the body's nose in degrees, + = Rex's RIGHT (the
# neck convention motion_agency uses). Registered by whoever owns the neck
# readback so this module needs no import from intelligence/. Only consulted
# when FLEX_DOA_MOUNT == "head".
_neck_yaw_provider: Optional[Callable[[], Optional[float]]] = None


def _num(name: str, default: float) -> float:
    try:
        return float(getattr(config, name, default))
    except (TypeError, ValueError):
        return float(default)


def _wrap180(deg: float) -> float:
    d = (float(deg) + 180.0) % 360.0
    return d - 180.0 if d != 0.0 else 180.0


def set_neck_yaw_provider(fn: Optional[Callable[[], Optional[float]]]) -> None:
    """Install the neck-yaw readback used for a head-mounted ring (deg, + = right)."""
    global _neck_yaw_provider
    _neck_yaw_provider = fn


def chip_to_base_bearing(doa_deg: float, neck_yaw_right_deg: Optional[float] = None) -> float:
    """Chip DoA (0-359) → base-frame bearing (deg, + = left/CCW, wrapped ±180).

    ``neck_yaw_right_deg`` is applied only for a head mount: a sound the head
    (turned θ to the right) hears dead ahead sits −θ in the base frame.
    """
    sign = _num("FLEX_DOA_SIGN", 1.0) or 1.0
    offset = _num("FLEX_DOA_FORWARD_OFFSET_DEG", 0.0)
    bearing = sign * (float(doa_deg) - offset)
    if str(getattr(config, "FLEX_DOA_MOUNT", "base")).strip().lower() == "head" \
            and neck_yaw_right_deg is not None:
        bearing -= float(neck_yaw_right_deg)
    return _wrap180(bearing)


def dominant_cluster(bearings: "list[float]", cluster_deg: float) -> "Optional[dict]":
    """The largest group of mutually-agreeing bearings.

    Returns {"bearing_deg", "n", "cluster_n", "share", "spread_deg"} or None
    when the list is empty. Each sample seeds a candidate cluster of every
    sample within ±cluster_deg of it; the biggest wins (ties → the candidate
    seeded latest, i.e. the freshest reading), and the cluster's circular mean
    is the answer.
    """
    if not bearings:
        return None
    n = len(bearings)
    best_members: "list[float]" = []
    for seed in bearings:                      # later seeds overwrite on ties (>=)
        members = [b for b in bearings if abs(_wrap180(b - seed)) <= cluster_deg]
        if len(members) >= len(best_members):
            best_members = members
    sx = sum(math.cos(math.radians(b)) for b in best_members)
    sy = sum(math.sin(math.radians(b)) for b in best_members)
    centre = _wrap180(math.degrees(math.atan2(sy, sx))) if (sx or sy) else best_members[0]
    spread = sum(abs(_wrap180(b - centre)) for b in best_members) / max(1, len(best_members))
    return {"bearing_deg": centre, "n": n, "cluster_n": len(best_members),
            "share": len(best_members) / n, "spread_deg": spread}


# ── poller ────────────────────────────────────────────────────────────────────

def _open() -> bool:
    global _dev
    try:
        from tools import flex_ctl
        dev = flex_ctl.open_device()
    except Exception as exc:
        _status["last_error"] = str(exc)
        return False
    with _dev_lock:
        _dev = dev
    _status["connected"] = True
    _log.info("[flex_doa] connected — polling DOA_VALUE at %.0f Hz (mount=%s, sign=%+.0f, offset=%.0f°)",
              _num("FLEX_DOA_POLL_HZ", 8.0), getattr(config, "FLEX_DOA_MOUNT", "base"),
              _num("FLEX_DOA_SIGN", 1.0), _num("FLEX_DOA_FORWARD_OFFSET_DEG", 0.0))
    return True


def _close() -> None:
    global _dev
    with _dev_lock:
        dev, _dev = _dev, None
    _status["connected"] = False
    if dev is not None:
        try:
            dev.close()
        except Exception:
            pass


def _base_moving() -> bool:
    """True while the drive base reports anything but idle — its turns rotate the
    ring under the sound field and its motors/sfx are a sound source of their own."""
    try:
        from hardware import motion
        st = str(motion.state() or "unknown").lower()
    except Exception:
        return False
    return st not in ("idle", "unknown", "")


def _poll_once() -> bool:
    global _last_moving_at
    with _dev_lock:
        dev = _dev
    if dev is None:
        return False
    try:
        doa, speech = dev.read("DOA_VALUE")
        try:
            energy = float(dev.read("AEC_SPENERGY_VALUES")[3])
        except Exception:
            energy = 0.0
    except Exception as exc:
        _status["errors"] += 1
        _status["last_error"] = str(exc)
        return False
    neck = None
    if _neck_yaw_provider is not None:
        try:
            neck = _neck_yaw_provider()
        except Exception:
            neck = None
    now = time.monotonic()
    bearing = chip_to_base_bearing(float(doa), neck)
    moving = _base_moving()
    if moving:
        _last_moving_at = now
    elif (now - _last_moving_at) < _num("FLEX_DOA_MOTION_SETTLE_SECS", 0.6):
        moving = True                      # still settling after a maneuver
    keep = _num("FLEX_DOA_HISTORY_SECS", 20.0)
    with _lock:
        _samples.append((now, float(doa), bearing, bool(speech), energy, moving))
        while _samples and (now - _samples[0][0]) > keep:
            _samples.popleft()
    _status["reads"] += 1
    return True


def _loop() -> None:
    period = 1.0 / max(1.0, _num("FLEX_DOA_POLL_HZ", 8.0))
    retry = max(5.0, _num("FLEX_DOA_RECONNECT_SECS", 30.0))
    fails = 0
    next_open = 0.0
    while not _stop.is_set():
        if not _status["connected"]:
            now = time.monotonic()
            if now >= next_open:
                if not _open():
                    next_open = now + retry
            if not _status["connected"]:
                _stop.wait(min(retry, 1.0))
                continue
        t = time.monotonic()
        if _poll_once():
            fails = 0
        else:
            fails += 1
            if fails >= 5:
                _log.warning("[flex_doa] %d consecutive read failures (%s) — reopening the device",
                             fails, _status["last_error"])
                _close()
                fails = 0
                next_open = time.monotonic() + 2.0
                continue
        _stop.wait(max(0.0, period - (time.monotonic() - t)))
    _close()


def start() -> bool:
    """Start the poller. Returns True when it is running (device present or
    watching for it). No-op when FLEX_DOA_ENABLED is off or no ReSpeaker Flex
    is the configured mic."""
    global _thread
    if not bool(getattr(config, "FLEX_DOA_ENABLED", True)):
        _log.info("[flex_doa] disabled (config.FLEX_DOA_ENABLED)")
        return False
    try:
        from utils.config_loader import AUDIO_DEVICE_INDEX
        import sounddevice as sd
        name = str(sd.query_devices(AUDIO_DEVICE_INDEX).get("name") or "").lower() \
            if AUDIO_DEVICE_INDEX is not None else ""
    except Exception:
        name = ""
    if "xvf3800" not in name and "flex" not in name:
        _log.info("[flex_doa] disabled — the configured mic is not a Flex XVF3800 (%r)", name)
        return False
    if _thread is not None and _thread.is_alive():
        return True
    _stop.clear()
    _status["enabled"] = True
    _thread = threading.Thread(target=_loop, name="flex-doa", daemon=True)
    _thread.start()
    return True


def stop() -> None:
    global _thread
    _stop.set()
    t = _thread
    if t is not None and t.is_alive() and t is not threading.current_thread():
        t.join(timeout=2.0)
    _thread = None
    _status["enabled"] = False
    _close()


def available() -> bool:
    return bool(_status["enabled"] and _status["connected"])


def status() -> dict:
    return dict(_status)


# ── queries ───────────────────────────────────────────────────────────────────

def bearing_between(t0: float, t1: float) -> "Optional[dict]":
    """Where the voice came from over [t0, t1] (monotonic seconds).

    Uses the speech-flagged samples in the window (padded by
    FLEX_DOA_SEGMENT_PAD_SECS on both sides), takes the dominant cluster, and
    returns {"bearing_deg" (base frame, + = left), "raw_deg", "n", "cluster_n",
    "share", "spread_deg", "t0", "t1"} — or None when the poller is off, the
    window holds fewer than FLEX_DOA_MIN_SAMPLES speech samples, or no cluster
    reaches FLEX_DOA_MIN_CLUSTER_SHARE of them.
    """
    if not available():
        return None
    pad = _num("FLEX_DOA_SEGMENT_PAD_SECS", 0.3)
    lo, hi = float(t0) - pad, float(t1) + pad
    with _lock:
        rows = [s for s in _samples if lo <= s[0] <= hi and s[3] and not (len(s) > 5 and s[5])]
    if len(rows) < int(_num("FLEX_DOA_MIN_SAMPLES", 3)):
        return None
    cluster = dominant_cluster([r[2] for r in rows], _num("FLEX_DOA_CLUSTER_DEG", 20.0))
    if cluster is None or cluster["share"] < _num("FLEX_DOA_MIN_CLUSTER_SHARE", 0.4):
        return None
    raw = dominant_cluster([r[1] for r in rows], _num("FLEX_DOA_CLUSTER_DEG", 20.0))
    cluster.update({"raw_deg": (raw["bearing_deg"] % 360.0) if raw else None,
                    "t0": float(t0), "t1": float(t1)})
    return cluster


def latest(max_age_secs: float = 1.0) -> "Optional[dict]":
    """The most recent speech-flagged sample, or None if older than max_age_secs."""
    with _lock:
        for s in reversed(_samples):
            if s[3]:
                if (time.monotonic() - s[0]) <= max_age_secs:
                    return {"t": s[0], "raw_deg": s[1], "bearing_deg": s[2], "energy": s[4]}
                return None
    return None


def _reset_for_tests() -> None:
    with _lock:
        _samples.clear()
    _status.update(enabled=False, connected=False, reads=0, errors=0, last_error="")


def _inject_for_tests(rows) -> None:
    """(t, doa_raw, base_bearing, speech, energy[, moving]) rows straight into the history."""
    with _lock:
        _samples.extend(tuple(r) if len(r) > 5 else tuple(r) + (False,) for r in rows)
    _status.update(enabled=True, connected=True)
