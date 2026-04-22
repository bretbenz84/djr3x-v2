"""
Single OpenCV camera stream with thread-safe frame buffer and auto-reconnect.

The camera is opened once at start() and never intentionally closed during normal
operation. If the device disconnects, the capture loop detects the failed read and
retries every CAMERA_RECONNECT_INTERVAL_SECS until the device comes back.

All public functions are no-ops when CAMERA_ENABLED is False or CAMERA_INDEX is None.
"""

import logging
import threading
import time
from typing import Optional

import numpy as np

import config
from utils.config_loader import CAMERA_ENABLED, CAMERA_INDEX

_log = logging.getLogger(__name__)

_cap = None          # cv2.VideoCapture — module-level singleton
_frame: Optional[np.ndarray] = None
_frame_lock = threading.Lock()
_stop_event = threading.Event()
_capture_thread: Optional[threading.Thread] = None


# ── Public API ────────────────────────────────────────────────────────────────

def start() -> None:
    """Open the camera and start the background capture thread."""
    global _capture_thread
    if not CAMERA_ENABLED:
        _log.debug("CAMERA_ENABLED=False — camera start is a no-op")
        return
    _stop_event.clear()
    _capture_thread = threading.Thread(
        target=_capture_loop,
        daemon=True,
        name="camera-capture",
    )
    _capture_thread.start()
    _log.info("Camera capture thread started (index=%d)", CAMERA_INDEX)


def stop() -> None:
    """Signal the capture thread to exit and release the camera."""
    if not CAMERA_ENABLED:
        return
    _stop_event.set()
    if _capture_thread is not None:
        _capture_thread.join(timeout=5.0)
    _close_camera()


def get_frame() -> Optional[np.ndarray]:
    """Return a copy of the most recent frame, or None if none available yet."""
    with _frame_lock:
        if _frame is None:
            return None
        return _frame.copy()


def capture_still() -> Optional[np.ndarray]:
    """
    High-quality single capture for face enrollment and vision queries.

    Raises the visor to its maximum position, centers the neck, waits
    CAMERA_POSE_SETTLE_SECS for servos to settle, then returns a frame.
    Servo positions are restored in a finally block regardless of outcome.

    Returns the captured frame or None if the camera is unavailable.
    """
    if not CAMERA_ENABLED:
        _log.debug("CAMERA_ENABLED=False — capture_still is a no-op")
        return None

    from hardware import servos

    visor_cfg = config.SERVO_CHANNELS["visor"]
    neck_cfg  = config.SERVO_CHANNELS["neck"]

    visor_before = servos.get_servo(visor_cfg["ch"]) or visor_cfg["neutral"]
    neck_before  = servos.get_servo(neck_cfg["ch"])  or neck_cfg["neutral"]

    try:
        servos.set_servos({
            visor_cfg["ch"]: visor_cfg["max"],
            neck_cfg["ch"]:  neck_cfg["neutral"],
        })
        time.sleep(config.CAMERA_POSE_SETTLE_SECS)
        frame = get_frame()
        if frame is None:
            _log.warning("capture_still: no frame available from buffer")
        return frame
    finally:
        servos.set_servos({
            visor_cfg["ch"]: visor_before,
            neck_cfg["ch"]:  neck_before,
        })


# ── Internal ──────────────────────────────────────────────────────────────────

def _open_camera() -> bool:
    """Open the VideoCapture device and apply resolution settings. Returns True on success."""
    global _cap
    import cv2
    cap = cv2.VideoCapture(CAMERA_INDEX)
    if not cap.isOpened():
        cap.release()
        _log.warning("Camera open failed (index=%d)", CAMERA_INDEX)
        return False
    cap.set(cv2.CAP_PROP_FRAME_WIDTH,  config.CAMERA_WIDTH)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, config.CAMERA_HEIGHT)
    _cap = cap
    _log.info(
        "Camera opened (index=%d, %dx%d)",
        CAMERA_INDEX, config.CAMERA_WIDTH, config.CAMERA_HEIGHT,
    )
    return True


def _close_camera() -> None:
    global _cap
    if _cap is not None:
        _cap.release()
        _cap = None
        _log.info("Camera closed")


def _capture_loop() -> None:
    """Daemon thread: reads frames continuously and stores the latest in the shared buffer."""
    global _frame

    if not _open_camera():
        _log.warning(
            "Initial camera open failed — will retry every %.1fs",
            config.CAMERA_RECONNECT_INTERVAL_SECS,
        )

    while not _stop_event.is_set():
        if _cap is None or not _cap.isOpened():
            _log.info(
                "Attempting camera reconnection (index=%d)", CAMERA_INDEX
            )
            _close_camera()
            _stop_event.wait(config.CAMERA_RECONNECT_INTERVAL_SECS)
            _open_camera()
            continue

        ret, frame = _cap.read()
        if not ret:
            _log.warning("Camera read failed — treating as disconnect (index=%d)", CAMERA_INDEX)
            _close_camera()
            continue

        with _frame_lock:
            _frame = frame

    _close_camera()
    _log.info("Camera capture thread stopped")
