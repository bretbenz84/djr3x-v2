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
_last_self_at: float = 0.0         # last poll that saw Rex playing sound (tail window anchor)
_last_yaw: Optional[float] = None  # base IMU yaw at the previous poll (rotation detector)
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


def heroarm_ring_yaw_deg(heroarm_qus: Optional[float]) -> float:
    """How far the ring's 0° is swung (deg, + = toward Rex's LEFT) by the hero-arm
    servo at ``heroarm_qus``: linear from neutral, FLEX_DOA_HEROARM_YAW_DEG_AT_MAX
    at the channel max, mirrored below neutral. 0 when unmeasured/unknown."""
    full = _num("FLEX_DOA_HEROARM_YAW_DEG_AT_MAX", 0.0)
    if not full or heroarm_qus is None:
        return 0.0
    try:
        cfg = config.SERVO_CHANNELS["heroarm"]
        neutral = float(cfg["neutral"]); hi = float(cfg["max"])
    except Exception:
        return 0.0
    span = max(1.0, hi - neutral)
    return full * (float(heroarm_qus) - neutral) / span


def current_neck_qus() -> Optional[float]:
    """Neck servo position at this poll (logged per bearing so head shadowing of
    the ring can be correlated with bad readings; not used in the math)."""
    try:
        from world_state import world_state
        positions = (world_state.get("self_state") or {}).get("servo_positions") or {}
        v = positions.get("neck")
        return float(v) if v is not None else None
    except Exception:
        return None


def current_heroarm_qus() -> Optional[float]:
    try:
        from world_state import world_state
        positions = (world_state.get("self_state") or {}).get("servo_positions") or {}
        v = positions.get("heroarm")
        return float(v) if v is not None else None
    except Exception:
        return None


def chip_to_base_bearing(doa_deg: float, neck_yaw_right_deg: Optional[float] = None,
                         heroarm_qus: Optional[float] = None) -> float:
    """Chip DoA (0-359) → base-frame bearing (deg, + = left/CCW, wrapped ±180).

    ``neck_yaw_right_deg`` is applied only for a head mount: a sound the head
    (turned θ to the right) hears dead ahead sits −θ in the base frame.
    ``heroarm_qus`` corrects for the hero-arm section the ring is mounted on
    (heroarm_ring_yaw_deg): a ring swung φ to the left reports a source φ
    further right than it is, so φ is added back.
    """
    sign = _num("FLEX_DOA_SIGN", 1.0) or 1.0
    offset = _num("FLEX_DOA_FORWARD_OFFSET_DEG", 0.0)
    bearing = sign * (float(doa_deg) - offset)
    if str(getattr(config, "FLEX_DOA_MOUNT", "base")).strip().lower() == "head" \
            and neck_yaw_right_deg is not None:
        bearing -= float(neck_yaw_right_deg)
    bearing += heroarm_ring_yaw_deg(heroarm_qus)
    return _wrap180(bearing)


def dominant_cluster(bearings: "list[float]", cluster_deg: float,
                     weights: "Optional[list[float]]" = None) -> "Optional[dict]":
    """The heaviest group of mutually-agreeing bearings.

    Returns {"bearing_deg", "n", "cluster_n", "share", "spread_deg", "weight",
    "weight_share"} or None when the list is empty. Each sample seeds a
    candidate cluster of every sample within ±cluster_deg of it; the cluster
    with the largest summed weight wins (ties → the candidate seeded latest,
    i.e. the freshest reading), and its weighted circular mean is the answer.
    Unweighted, weight = count. With the chip's speech energy as the weight
    (owner observation 2026-09-02: right bearings came with high energy, wrong
    ones with low — reflections), a few strong direct-path samples outvote a
    pile of weak ones.
    """
    if not bearings:
        return None
    n = len(bearings)
    if weights is None or len(weights) != n:
        weights = [1.0] * n
    weights = [max(0.0, float(w)) for w in weights]
    total_w = sum(weights) or float(n)
    best_idx: "list[int]" = []
    best_w = -1.0
    for i in range(n):                          # later seeds overwrite on ties (>=)
        idx = [j for j in range(n) if abs(_wrap180(bearings[j] - bearings[i])) <= cluster_deg]
        w = sum(weights[j] for j in idx)
        if w >= best_w:
            best_w, best_idx = w, idx
    ww = [weights[j] if weights[j] > 0 else 1e-9 for j in best_idx]
    sx = sum(w * math.cos(math.radians(bearings[j])) for w, j in zip(ww, best_idx))
    sy = sum(w * math.sin(math.radians(bearings[j])) for w, j in zip(ww, best_idx))
    centre = _wrap180(math.degrees(math.atan2(sy, sx))) if (sx or sy) else bearings[best_idx[0]]
    spread = sum(abs(_wrap180(bearings[j] - centre)) for j in best_idx) / max(1, len(best_idx))
    return {"bearing_deg": centre, "n": n, "cluster_n": len(best_idx),
            "share": len(best_idx) / n, "spread_deg": spread,
            "weight": best_w, "weight_share": (best_w / total_w) if total_w > 0 else 0.0}


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
    """True while the base is ROTATING — its turns swing the ring under the sound
    field. Judged from the base's own gyro yaw when telemetry carries one (a
    yaw step over FLEX_DOA_MOTION_YAW_STEP_DEG since the previous poll); the
    coarse `state != idle` gate only when there is no IMU, because that gate
    blanked whole windows around the idle wander's 5° sways."""
    global _last_yaw
    try:
        from hardware import motion
        tele = motion.telemetry()
    except Exception:
        return False
    if not isinstance(tele, dict):
        return False
    imu = tele.get("imu")
    yaw = None
    if isinstance(imu, dict) and imu.get("ok") and imu.get("yaw") is not None:
        try:
            yaw = float(imu["yaw"])
        except (TypeError, ValueError):
            yaw = None
    if yaw is not None:
        prev, _last_yaw = _last_yaw, yaw
        if prev is None:
            return False
        return abs(_wrap180(yaw - prev)) > _num("FLEX_DOA_MOTION_YAW_STEP_DEG", 1.0)
    st = str(tele.get("state") or "unknown").lower()
    return st not in ("idle", "unknown", "")


def _self_speaking() -> bool:
    """True while Rex's own audio is playing (TTS, sound effects, wake ack): the
    ring hears the speaker as a talker and the chip tags it as speech."""
    try:
        from audio import speech_queue
        if speech_queue.is_speaking():
            return True
    except Exception:
        pass
    try:
        from audio import output_gate
        if output_gate.is_busy():
            return True
    except Exception:
        pass
    try:
        from audio import echo_cancel
        if echo_cancel.is_suppressed():
            return True
    except Exception:
        pass
    return False


def _poll_once() -> bool:
    global _last_moving_at, _last_self_at
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
        beam_deg = None
        try:
            az = dev.read("AEC_AZIMUTH_VALUES")
            if len(az) >= 4:
                beam_deg = math.degrees(float(az[3])) % 360.0
        except Exception:
            beam_deg = None
    except Exception as exc:
        _status["errors"] += 1
        _status["last_error"] = str(exc)
        return False
    # The beam azimuth leads the DoA register by ~1 s when the talker moved;
    # trust it while the chip reports speech energy on it.
    raw_deg = float(doa)
    speech_flag = bool(speech)
    if beam_deg is not None and energy >= _num("FLEX_DOA_BEAM_ENERGY_MIN", 50000.0):
        raw_deg = float(beam_deg)
        speech_flag = True
    neck = None
    if _neck_yaw_provider is not None:
        try:
            neck = _neck_yaw_provider()
        except Exception:
            neck = None
    now = time.monotonic()
    hero = current_heroarm_qus()
    neck_qus = current_neck_qus()
    bearing = chip_to_base_bearing(raw_deg, neck, hero)
    moving = _base_moving()
    if moving:
        _last_moving_at = now
    elif (now - _last_moving_at) < _num("FLEX_DOA_MOTION_SETTLE_SECS", 0.6):
        moving = True                      # still settling after a maneuver
    if _self_speaking():
        _last_self_at = now
        moving = True                      # Rex's own voice: not a talker
    elif (now - _last_self_at) < _num("FLEX_DOA_SELF_SPEECH_TAIL_SECS", 0.4):
        moving = True                      # the room is still ringing with him
    keep = _num("FLEX_DOA_HISTORY_SECS", 20.0)
    with _lock:
        _samples.append((now, raw_deg, bearing, speech_flag, energy, moving, hero, neck_qus))
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
    min_n = int(_num("FLEX_DOA_MIN_SAMPLES", 3))
    if len(rows) < min_n:
        return None
    cluster_deg = _num("FLEX_DOA_CLUSTER_DEG", 20.0)
    energies = [max(0.0, float(r[4])) for r in rows]
    if any(e > 0.0 for e in energies):
        # ENERGY-WEIGHTED vote over the whole window: the direct path carries
        # the speech energy, reflections and the chip's stale hold come in
        # weak (owner observation 2026-09-02). A zero-energy sample still
        # counts a little so a window of DOA-only samples is not empty.
        floor_w = max(1.0, 0.02 * max(energies))
        # Recency rides on top (0.5 at the window's start → 1.0 at its end): the
        # chip's direction register lags a talker who moved, so with FLAT
        # energies the converged tail still outvotes the stale head.
        t_first, t_last = rows[0][0], rows[-1][0]
        span = max(1e-6, t_last - t_first)
        weights = [max(e, floor_w) * (0.5 + 0.5 * (r[0] - t_first) / span)
                   for e, r in zip(energies, rows)]
        pool = rows
        cluster = dominant_cluster([r[2] for r in pool], cluster_deg, weights)
        ok = cluster is not None and cluster["weight_share"] >= _num("FLEX_DOA_MIN_CLUSTER_SHARE", 0.4)
        raw = dominant_cluster([r[1] for r in pool], cluster_deg, weights)
    else:
        # No energy readings at all: the chip's direction register lags a talker
        # who moved, so decide from the TAIL of the phrase (the converged part).
        tail_n = max(min_n, int(_num("FLEX_DOA_TAIL_MIN_SAMPLES", 5)),
                     int(round(len(rows) * _num("FLEX_DOA_TAIL_SHARE", 0.5))))
        pool = rows[-tail_n:]
        cluster = dominant_cluster([r[2] for r in pool], cluster_deg)
        ok = cluster is not None and cluster["share"] >= _num("FLEX_DOA_MIN_CLUSTER_SHARE", 0.4)
        raw = dominant_cluster([r[1] for r in pool], cluster_deg)
    if not ok:
        return None
    cluster["n"] = len(rows)            # n = every speech sample in the window
    whole = dominant_cluster([r[2] for r in rows], cluster_deg)
    heroes = [r[6] for r in rows if len(r) > 6 and r[6] is not None]
    necks = [r[7] for r in rows if len(r) > 7 and r[7] is not None]
    cluster.update({
        "heroarm_qus": (sum(heroes) / len(heroes)) if heroes else None,
        "neck_qus": (sum(necks) / len(necks)) if necks else None,
        "raw_deg": (raw["bearing_deg"] % 360.0) if raw else None,
        "t0": float(t0), "t1": float(t1),
        "window_n": len(rows),
        "tail_n": len(pool),
        "clusters": cluster_summary([r[2] for r in rows], cluster_deg, energies=energies),
        "head_disagrees": bool(whole and abs(_wrap180(whole["bearing_deg"] - cluster["bearing_deg"])) > cluster_deg),
    })
    return cluster


def cluster_summary(bearings: "list[float]", cluster_deg: float, top: int = 3,
                    energies: "Optional[list[float]]" = None) -> "list[tuple]":
    """[(centre_deg, count, mean_energy), ...] of the largest distinct groups,
    heaviest first, for the logs — '−131°×6 e=0.9M, +154°×14 e=0.1M' says at a
    glance that a weak reflection outnumbered the talker."""
    remaining = list(range(len(bearings)))
    out = []
    while remaining and len(out) < top:
        bs = [bearings[i] for i in remaining]
        ws = [max(1.0, float(energies[i])) for i in remaining] if energies else None
        c = dominant_cluster(bs, cluster_deg, ws)
        if c is None:
            break
        members = [i for i in remaining if abs(_wrap180(bearings[i] - c["bearing_deg"])) <= cluster_deg]
        mean_e = (sum(float(energies[i]) for i in members) / len(members)) if (energies and members) else 0.0
        out.append((c["bearing_deg"], len(members), mean_e))
        remaining = [i for i in remaining if i not in members]
    return out


def describe_clusters(res: "Optional[dict]") -> str:
    if not res or not res.get("clusters"):
        return ""
    parts = []
    for g in res["clusters"]:
        c, k = g[0], g[1]
        e = g[2] if len(g) > 2 else 0.0
        parts.append(f"{c:+.0f}°×{k}" + (f" e={e / 1e6:.2f}M" if e else ""))
    return ", ".join(parts)


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
    """(t, doa_raw, base_bearing, speech, energy[, moving[, heroarm]]) rows straight into the history."""
    with _lock:
        for r in rows:
            r = tuple(r)
            while len(r) < 8:
                r = r + ((False,) if len(r) == 5 else (None,))
            _samples.append(r)
    _status.update(enabled=True, connected=True)
