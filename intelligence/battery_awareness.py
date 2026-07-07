"""
intelligence/battery_awareness.py — Rex knows how his power cells feel.

Reads pack voltage from the motion base's telemetry (INA226 on the ESP32's I2C
bus; firmware sends batt_mv = -1 when no sensor is wired, so this module stays
completely dormant until the hardware exists).

The pack is a 12.8V 4S LiFePO4 — a chemistry with a famously FLAT discharge
curve: it rests near 13.0-13.2V from ~90% down to ~25%, then falls off a knee.
Voltage can honestly distinguish only a few bands, so that is exactly what this
module claims: CHARGING/FULL, NOMINAL (the long flat middle), LOW (~20%), and
CRITICAL (~10%, near the BMS's own cutoff). No fake percentages.

Behavior:
  - One in-character grumble per DOWNWARD tier crossing per session, spoken only
    when someone is visibly present to hear it (a latched crossing waits for the
    next co-presence). 100mV hysteresis so a sagging pack can't flap tiers.
  - motion_agency consults battery_critical() and declines spontaneous
    approaches when the pack is critical (voice-commanded drives still obey —
    the BMS protects the pack; Rex just stops volunteering).
"""

import logging
import time
from typing import Optional

import config

_log = logging.getLogger(__name__)

# Tier order matters: index 0 is the healthiest. (name, floor_mv)
_TIERS = (
    ("charging", 13450),   # >= 13.45V at rest only happens on/near the charger
    ("nominal", 12950),    # the long LiFePO4 plateau (~25-90%)
    ("low", 12450),        # ~12.5-12.95V — the knee, roughly 10-25%
    ("critical", 0),       # below the knee; BMS cutoff isn't far away
)

_last_tier: Optional[str] = None
_announced_tiers: set = set()
_pending_announce: Optional[str] = None
_last_read_mv: int = -1
_last_spoke_at: float = 0.0


def _lines_for(tier: str) -> list[str]:
    lines = getattr(config, "BATTERY_TIER_LINES", {}) or {}
    return list(lines.get(tier, []) or [])


def current_mv() -> int:
    """Latest pack millivolts from telemetry, or -1 when unknown/no sensor."""
    try:
        from hardware import motion
        snap = motion.telemetry() or {}
        mv = int(snap.get("batt_mv", -1) or -1)
        return mv if mv > 0 else -1
    except Exception:
        return -1


def tier_for_mv(mv: int, *, previous: Optional[str] = None) -> Optional[str]:
    """Map millivolts to a tier with 100mV hysteresis against the previous tier
    (a pack sagging under drive load must not flap low<->nominal)."""
    if mv <= 0:
        return None
    hysteresis = int(getattr(config, "BATTERY_TIER_HYSTERESIS_MV", 100) or 0)
    for i, (name, floor) in enumerate(_TIERS):
        if mv >= floor:
            # Moving UP a tier requires clearing the floor by the hysteresis.
            if previous is not None and previous != name:
                prev_idx = next(
                    (j for j, (n, _f) in enumerate(_TIERS) if n == previous), None
                )
                if prev_idx is not None and i < prev_idx and mv < floor + hysteresis:
                    return previous
            return name
    return _TIERS[-1][0]


def battery_critical() -> bool:
    """True when the pack is in the critical band — motion_agency stops
    volunteering drives. False when unknown (no sensor = no opinion)."""
    if not bool(getattr(config, "BATTERY_AWARENESS_ENABLED", True)):
        return False
    return tier_for_mv(current_mv()) == "critical"


def step(snapshot: dict, profile) -> None:
    """One consciousness tick: track tier crossings, grumble once per downward
    crossing when someone is present. Never raises."""
    global _last_tier, _pending_announce, _last_read_mv, _last_spoke_at
    try:
        if not bool(getattr(config, "BATTERY_AWARENESS_ENABLED", True)):
            return
        mv = current_mv()
        if mv <= 0:
            return
        _last_read_mv = mv
        tier = tier_for_mv(mv, previous=_last_tier)
        if tier is None:
            return
        if _last_tier is None:
            _last_tier = tier      # first reading of the session: baseline, no remark
            return
        if tier != _last_tier:
            order = [name for name, _f in _TIERS]
            downward = order.index(tier) > order.index(_last_tier)
            _log.info(
                "[battery] tier %s -> %s (%.2fV)", _last_tier, tier, mv / 1000.0
            )
            _last_tier = tier
            if downward and tier in ("low", "critical") and tier not in _announced_tiers:
                _pending_announce = tier

        if _pending_announce is None:
            return
        # Speak only with an audience; a latched crossing waits for co-presence.
        people = snapshot.get("people") or []
        someone_here = any(
            isinstance(p, dict) and (p.get("face_visible") or p.get("face_box"))
            for p in people
        )
        if not someone_here:
            return
        if getattr(profile, "user_mid_sentence", False) or getattr(
            profile, "interaction_busy", False
        ):
            return
        cooldown = float(getattr(config, "BATTERY_ANNOUNCE_MIN_GAP_SECS", 300.0) or 0.0)
        now = time.monotonic()
        if cooldown and (now - _last_spoke_at) < cooldown:
            return
        lines = _lines_for(_pending_announce)
        if not lines:
            _pending_announce = None
            return
        import random
        from intelligence import speech_engine
        tier = _pending_announce
        spoke = speech_engine.speak_async(
            random.choice(lines),
            emotion=("worried" if tier == "critical" else "neutral"),
            purpose="battery_status",
            label=f"battery {tier}",
        )
        if spoke:
            _announced_tiers.add(tier)
            _pending_announce = None
            _last_spoke_at = now
    except Exception as exc:
        _log.debug("battery awareness step error: %s", exc)
