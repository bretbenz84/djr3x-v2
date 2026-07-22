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
    try:
        recognizer = PlaceRecognizer(
            embed_fn=embedder.encode_image,
            get_heading=_get_heading,
            get_motion_state=_get_motion_state,
            get_person_occlusion=_get_person_occlusion,
            world_state=world_state,
            emit_event=emit_event or _default_emit,
            model_tag=embedder.model_tag,
        )
    except Exception as exc:  # noqa: BLE001
        _log.warning("place recognition failed to initialize (%s); feature off", exc)
        return
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

def _get_heading():
    try:
        from hardware import compass
        return compass.get_service_yaw()        # float [0,360) or None (service off/no fix)
    except Exception:
        return None


def _get_motion_state():
    from perception.place_recognition import MotionState
    try:
        from hardware import motion
        t = motion.telemetry() or {}
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
        return MotionState()


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
