"""
perception/place_service.py — runtime wiring for visual place recognition.

Module-level singleton (mirrors ``vision.camera`` / ``vision.pose``): ``start()`` loads the
MobileCLIP-S2 encoder, constructs the ``PlaceRecognizer`` with real signal adapters, and
runs a small daemon thread that feeds camera frames at ``PLACE_OBSERVE_INTERVAL_S``. The
recognizer publishes its debounced belief to ``world_state.current_place`` on its own; this
service just supplies frames + signals and exposes the enrollment API upward.

Everything is fail-safe and lazy: heavy deps (torch/open_clip) load inside ``start()``, so
importing this module is cheap, and any failure (no encoder, no camera) degrades to
"feature off" — the rest of Rex is untouched and ``world_state.current_place`` stays None.

The conversation layer drives enrollment through the passthroughs (``enroll`` /
``cancel_enrollment`` / ``confirm_duplicate``) after it parses intent — this module never
parses language. Edge events (unknown_place, possible_duplicate_place, place_enrolled,
enrollment_failed) go to the injected ``emit_event`` sink; today that logs, and a
consciousness ``_step_`` can later poll/claim a proactive purpose off the same seam.
"""

from __future__ import annotations

import logging
import threading

import config

_log = logging.getLogger(__name__)

_lock = threading.Lock()
_recognizer = None
_embedder = None
_thread = None
_stop = threading.Event()


def enabled() -> bool:
    return bool(getattr(config, "PLACE_RECOGNITION_ENABLED", False))


# ── Lifecycle ────────────────────────────────────────────────────────────────────

def start(world_state=None, emit_event=None) -> bool:
    """Start place recognition. Returns immediately: the ~10 s MobileCLIP load happens
    inside the worker thread so it never blocks app startup. Returns False only when the
    feature is disabled by config; a later encoder/camera failure just logs and the thread
    exits (feature off). Idempotent — a second call while running is a no-op True."""
    global _thread
    if not enabled():
        _log.info("place recognition disabled (PLACE_RECOGNITION_ENABLED=False)")
        return False
    with _lock:
        if _thread is not None and _thread.is_alive():
            return True
        _stop.clear()
        _thread = threading.Thread(
            target=_run, args=(world_state, emit_event),
            name="place-recognition", daemon=True,
        )
        _thread.start()
        return True


def stop() -> None:
    global _recognizer, _embedder, _thread
    _stop.set()
    t = _thread
    if t is not None:
        t.join(timeout=2.0)
    with _lock:
        if _recognizer is not None:
            try:
                _recognizer.close()
            except Exception:
                pass
        _recognizer = None
        _embedder = None
        _thread = None


def _run(world_state, emit_event) -> None:
    """Worker thread: build the encoder + recognizer (slow), then feed frames until stop."""
    global _recognizer, _embedder
    from perception.place_embedder import load_place_embedder
    embedder = load_place_embedder()
    if embedder is None:
        return  # load_place_embedder already logged why; feature stays off
    from perception.place_recognition import PlaceRecognizer
    if world_state is None:
        from world_state import world_state as _ws_singleton
        world_state = _ws_singleton
    sink = emit_event or _default_emit
    holder: dict = {}

    def _emit(name, payload):
        try:
            sink(name, payload)
        except Exception as exc:
            _log.debug("place event sink failed for %s: %s", name, exc)
        if name == "possible_duplicate_place":
            # Voice enrollment always carries an explicit human-given name. If the new
            # room merely LOOKS like a known one, trust the human: a different name
            # means a different room — commit it as its own place. (A same-name tell
            # never reaches here; enroll() attaches to the existing row.) Without this,
            # CONFIRMING would wait forever for a confirm_duplicate() nobody sends and
            # the told room would silently never commit. Called synchronously on the
            # emitting thread — the recognizer lock is re-entrant.
            rec = holder.get("rec")
            if rec is not None:
                _log.info(
                    "auto-resolving duplicate: %r resembles %r (sim=%.2f) but was "
                    "told by name — keeping it as its own room",
                    payload.get("new_place"), payload.get("existing_place"),
                    float(payload.get("similarity") or 0.0),
                )
                try:
                    rec.confirm_duplicate(False)
                except Exception as exc:
                    _log.warning("duplicate auto-resolve failed: %s", exc)

    try:
        recognizer = PlaceRecognizer(
            embed_fn=embedder.encode_image,
            get_heading=_get_heading,
            get_motion_state=_get_motion_state,
            get_person_occlusion=_get_person_occlusion,
            world_state=world_state,
            emit_event=_emit,
            model_tag=embedder.model_tag,
        )
    except Exception as exc:  # noqa: BLE001
        _log.warning("place recognition failed to initialize (%s); feature off", exc)
        return
    holder["rec"] = recognizer
    if _stop.is_set():        # a stop() arrived during the slow load
        recognizer.close()
        return
    with _lock:
        _embedder = embedder
        _recognizer = recognizer
    _log.info("place recognition started (encoder=%s, tag=%s, places=%s)",
              embedder.name, embedder.model_tag, recognizer.place_names())

    from vision import camera
    interval = float(getattr(config, "PLACE_OBSERVE_INTERVAL_S", 1.5))
    while not _stop.wait(interval):
        try:
            frame = camera.get_frame()
            if frame is None:
                continue
            recognizer.observe(frame)   # embedder converts BGR->RGB; observe() self-throttles
            recognizer.tick()           # let enrollment timeouts fire during camera lulls
        except Exception as exc:  # a bad frame/encoder call must never kill the loop
            _log.debug("place observe tick failed: %s", exc)


# ── Signal adapters (all fail-safe: a dead sensor yields None / neutral) ──────────

def _neck_pan_deg():
    """Camera pan contributed by the NECK servo, in degrees around neutral. Head pans
    change the camera view exactly like chassis heading does, so enrollment view
    diversity credits them. None when no live servo reading exists (positions still at
    the world_state defaults because no Maestro is attached is indistinguishable from
    'head at neutral' — that's fine: a constant value just defers to the time gate)."""
    try:
        from world_state import world_state
        pos = ((world_state.get("self_state") or {}).get("servo_positions") or {}).get("neck")
        if pos is None:
            return None
        spec = (getattr(config, "SERVO_CHANNELS", {}) or {}).get("neck") or {}
        lo, hi = float(spec.get("min", 1984)), float(spec.get("max", 8960))
        neutral = float(spec.get("neutral", (lo + hi) / 2.0))
        if hi <= lo:
            return None
        span = float(getattr(config, "PLACE_NECK_SPAN_DEG", 120.0))
        return (float(pos) - neutral) / (hi - lo) * span
    except Exception:
        return None


def _get_heading():
    """Camera direction = chassis heading (compass) + head pan (neck servo). Either
    alone still provides useful capture diversity; None only when both are missing."""
    yaw = None
    try:
        from hardware import compass
        yaw = compass.get_service_yaw()         # float [0,360) or None (service off/no fix)
    except Exception:
        yaw = None
    neck = _neck_pan_deg()
    if yaw is None and neck is None:
        return None
    return ((yaw or 0.0) + (neck or 0.0)) % 360.0


def _get_motion_state():
    """MotionState from drive telemetry, or None when NO trustworthy signal exists (no
    base configured, or its telemetry stream is down). None disables the recognizer's
    freeze gate — with no sensor, 'not moving' can't be distinguished from 'was just
    carried to another room', and a pinned stale belief is the worse failure."""
    from perception.place_recognition import MotionState
    try:
        from utils.config_loader import MOTION_ESP32_PORT
        if not bool(getattr(config, "MOTION_ENABLED", True)) or not MOTION_ESP32_PORT:
            return None
        from hardware import motion
        t = motion.telemetry()
        if not t:
            return None
        wheels = t.get("wheels") or {}
        odom = t.get("odom") or {}
        wheels_moving = (
            t.get("state") == "moving"
            or abs(wheels.get("vl") or 0.0) > 1e-3
            or abs(wheels.get("vr") or 0.0) > 1e-3
        )
        # No raw-accel field in telemetry; the honest proxy is "IMU alive AND odometry
        # shows the chassis actually translating/rotating" (not head-servo jitter).
        accel_active = bool((t.get("imu") or {}).get("ok")) and (
            abs(odom.get("lin") or 0.0) > 1e-3 or abs(odom.get("ang") or 0.0) > 1e-3
        )
        return MotionState(wheels_moving=bool(wheels_moving), accel_active=bool(accel_active))
    except Exception:
        return None


def _get_person_occlusion() -> float:
    """Largest person's normalized body-bbox area from MediaPipe pose keypoints (already
    normalized to [0,1], so the span IS the frame fraction). 0.0 when nobody is tracked."""
    try:
        from world_state import world_state
        best = 0.0
        for person in (world_state.get("people") or []):
            kp = person.get("pose_keypoints") or {}
            pts = [v for v in kp.values() if v and len(v) >= 3 and v[2] >= 0.4]
            if len(pts) >= 2:
                xs = [v[0] for v in pts]
                ys = [v[1] for v in pts]
                best = max(best, (max(xs) - min(xs)) * (max(ys) - min(ys)))
        return float(best)
    except Exception:
        return 0.0


def _default_emit(name: str, payload: dict) -> None:
    _log.info("place event: %s %s", name, payload)


# ── Enrollment passthroughs for the conversation layer ───────────────────────────

def get_recognizer():
    return _recognizer


def enroll(name: str):
    rec = _recognizer
    return rec.enroll(name) if rec is not None else None


def cancel_enrollment() -> None:
    rec = _recognizer
    if rec is not None:
        rec.cancel_enrollment()


def confirm_duplicate(is_same: bool) -> bool:
    rec = _recognizer
    return rec.confirm_duplicate(is_same) if rec is not None else False


def current_place():
    rec = _recognizer
    return rec.current_place() if rec is not None else None


def state() -> str:
    rec = _recognizer
    return rec.state if rec is not None else "off"


def enrolling_name():
    rec = _recognizer
    return rec.enrolling_name() if rec is not None else None


def belief_context():
    rec = _recognizer
    return rec.belief_context() if rec is not None else None


def place_names() -> list:
    rec = _recognizer
    return rec.place_names() if rec is not None else []


def set_no_drive(name: str, on: bool, reason: "str | None" = None) -> bool:
    """Persist "don't drive in <room>". False when the room isn't enrolled (or the
    recognizer is off), which the caller should say out loud."""
    rec = _recognizer
    return bool(rec.set_no_drive(name, on, reason)) if rec is not None else False


def no_drive_places() -> dict:
    rec = _recognizer
    return rec.no_drive_places() if rec is not None else {}


def reject_belief(name: "str | None" = None) -> bool:
    """A human contradicted the believed room — drop it. True when one was dropped."""
    rec = _recognizer
    return bool(rec.reject_belief(name)) if rec is not None else False
