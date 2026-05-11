"""
Single camera stream with thread-safe frame buffer and auto-reconnect.

The camera is opened once at start() and never intentionally closed during normal
operation. If the device disconnects, the capture loop detects the failed read and
retries every CAMERA_RECONNECT_INTERVAL_SECS until the device comes back.

Robot mode uses OpenCV with CAMERA_INDEX. macOS dev mode can instead set
CAMERA_DEVICE_NAME to use ffmpeg + AVFoundation by device name, which avoids
whichever Continuity Camera macOS decides should be camera index 0 that day.

All public functions are no-ops when CAMERA_ENABLED is False.
"""

import logging
import os
import re
import shutil
import subprocess
import threading
import time
from typing import Callable, Optional

import numpy as np

import config
from utils.config_loader import (
    CAMERA_DEVICE_NAME,
    CAMERA_ENABLED,
    CAMERA_INDEX,
    CAMERA_SELECTION_DESCRIPTION,
)

_log = logging.getLogger(__name__)

_cap = None          # cv2.VideoCapture — module-level singleton
_frame: Optional[np.ndarray] = None
_last_frame_at: Optional[float] = None
_frame_lock = threading.Lock()
_stop_event = threading.Event()
_capture_thread: Optional[threading.Thread] = None
_reconnect_callbacks: list[Callable[[float], None]] = []
_reconnect_lock = threading.Lock()
_offline_since: Optional[float] = None


class _FFmpegCapture:
    """Minimal VideoCapture-like wrapper for ffmpeg AVFoundation capture."""

    def __init__(self, device_name: str):
        self._width = config.CAMERA_WIDTH
        self._height = config.CAMERA_HEIGHT
        self._frame_bytes = self._width * self._height * 3
        self._resolved_name = _resolve_avfoundation_device_name(device_name)

        cmd = [
            "ffmpeg",
            "-hide_banner",
            "-loglevel",
            "warning",
            "-f",
            "avfoundation",
            "-pixel_format",
            str(getattr(config, "CAMERA_AVFOUNDATION_PIXEL_FORMAT", "uyvy422") or "uyvy422"),
            "-framerate",
            str(config.CAMERA_FPS),
            "-video_size",
            f"{self._width}x{self._height}",
            "-i",
            f"{self._resolved_name}:none",
            "-an",
            "-fps_mode",
            "passthrough",
            "-pix_fmt",
            "bgr24",
            "-vcodec",
            "rawvideo",
            "-f",
            "rawvideo",
            "pipe:1",
        ]
        self._process = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            bufsize=0,
        )
        self._stderr_thread = threading.Thread(
            target=self._drain_stderr,
            daemon=True,
            name="camera-ffmpeg-stderr",
        )
        self._stderr_thread.start()

    @property
    def device_label(self) -> str:
        return self._resolved_name

    def isOpened(self) -> bool:
        return self._process.poll() is None and self._process.stdout is not None

    def read(self) -> tuple[bool, Optional[np.ndarray]]:
        if not self.isOpened() or self._process.stdout is None:
            return False, None

        frame_bytes = self._read_exactly(self._frame_bytes)
        if frame_bytes is None:
            return False, None

        frame = np.frombuffer(frame_bytes, dtype=np.uint8).reshape(
            (self._height, self._width, 3)
        )
        return True, frame

    def release(self) -> None:
        if self._process.poll() is None:
            self._process.terminate()
            try:
                self._process.wait(timeout=2.0)
            except subprocess.TimeoutExpired:
                self._process.kill()
                self._process.wait(timeout=2.0)

        if self._process.stdout is not None:
            self._process.stdout.close()
        if self._process.stderr is not None:
            self._process.stderr.close()

    def _drain_stderr(self) -> None:
        if self._process.stderr is None:
            return

        try:
            for raw_line in self._process.stderr:
                line = raw_line.decode("utf-8", errors="replace").strip()
                if not line:
                    continue
                _log.warning("ffmpeg camera: %s", line)
        except (OSError, ValueError):
            return

    def _read_exactly(self, size: int) -> Optional[bytes]:
        chunks = bytearray()
        while len(chunks) < size:
            if self._process.poll() is not None or self._process.stdout is None:
                return None
            chunk = self._process.stdout.read(size - len(chunks))
            if not chunk:
                return None
            chunks.extend(chunk)
        return bytes(chunks)


# ── Public API ────────────────────────────────────────────────────────────────

def start() -> None:
    """Open the camera and start the background capture thread."""
    global _capture_thread, _frame, _last_frame_at, _offline_since
    if not CAMERA_ENABLED:
        _log.debug("CAMERA_ENABLED=False — camera start is a no-op")
        return
    with _frame_lock:
        _frame = None
        _last_frame_at = None
    with _reconnect_lock:
        _offline_since = None
    _stop_event.clear()
    _capture_thread = threading.Thread(
        target=_capture_loop,
        daemon=True,
        name="camera-capture",
    )
    _capture_thread.start()
    _log.info("Camera capture thread started (%s)", CAMERA_SELECTION_DESCRIPTION)


def stop() -> None:
    """Signal the capture thread to exit and release the camera."""
    if not CAMERA_ENABLED:
        return
    _stop_event.set()
    if _capture_thread is not None:
        _capture_thread.join(timeout=5.0)
    _close_camera()


def register_on_reconnect(callback: Callable[[float], None]) -> None:
    """Register a callback fired once a fresh frame arrives after an outage."""
    with _reconnect_lock:
        if callback not in _reconnect_callbacks:
            _reconnect_callbacks.append(callback)


def unregister_on_reconnect(callback: Callable[[float], None]) -> None:
    """Remove a previously registered reconnect callback."""
    with _reconnect_lock:
        try:
            _reconnect_callbacks.remove(callback)
        except ValueError:
            pass


def get_frame() -> Optional[np.ndarray]:
    """Return a copy of the most recent frame, or None if none available yet."""
    with _frame_lock:
        if _frame is None:
            return None
        return _frame.copy()


def has_recent_frame(max_age_secs: float = 2.0) -> bool:
    """Return True when a camera frame has arrived recently."""
    if not CAMERA_ENABLED:
        return False
    max_age = max(0.0, float(max_age_secs))
    with _frame_lock:
        if _frame is None or _last_frame_at is None:
            return False
        return (time.monotonic() - _last_frame_at) <= max_age


def wait_for_frame(timeout_secs: float = 2.0) -> bool:
    """Wait briefly for the first live camera frame after startup."""
    if not CAMERA_ENABLED:
        return False
    deadline = time.monotonic() + max(0.0, float(timeout_secs))
    while not _stop_event.is_set():
        if has_recent_frame(max_age_secs=max(1.0, float(timeout_secs) + 0.5)):
            return True
        if time.monotonic() >= deadline:
            return False
        time.sleep(0.05)
    return False


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


def capture_current_gaze(settle_secs: float = 0.15) -> Optional[np.ndarray]:
    """
    Capture a frame from Rex's current head/gaze direction.

    This opens the visor but does not center or restore the neck. It is intended
    for directed attention commands like "look left" where the caller has
    already moved the head and wants the image from that pose.
    """
    if not CAMERA_ENABLED:
        _log.debug("CAMERA_ENABLED=False — capture_current_gaze is a no-op")
        return None

    from hardware import servos

    visor_cfg = config.SERVO_CHANNELS["visor"]
    visor_before = servos.get_servo(visor_cfg["ch"]) or visor_cfg["neutral"]

    try:
        servos.set_servo(visor_cfg["ch"], visor_cfg["max"])
        time.sleep(max(0.0, float(settle_secs)))
        frame = get_frame()
        if frame is None:
            _log.warning("capture_current_gaze: no frame available from buffer")
        return frame
    finally:
        servos.set_servo(visor_cfg["ch"], visor_before)


# ── Internal ──────────────────────────────────────────────────────────────────

def _open_camera() -> bool:
    """Open the VideoCapture device and apply resolution settings. Returns True on success."""
    global _cap
    import cv2

    if CAMERA_DEVICE_NAME:
        if os.uname().sysname != "Darwin":
            _log.warning(
                "CAMERA_DEVICE_NAME=%r is only supported on macOS — camera open failed",
                CAMERA_DEVICE_NAME,
            )
            return False
        if shutil.which("ffmpeg") is None:
            _log.warning(
                "CAMERA_DEVICE_NAME=%r requires ffmpeg in PATH — camera open failed",
                CAMERA_DEVICE_NAME,
            )
            return False
        cap = _FFmpegCapture(CAMERA_DEVICE_NAME)
        time.sleep(0.25)
        if not cap.isOpened():
            cap.release()
            _log.warning("Camera open failed (device name match=%r)", CAMERA_DEVICE_NAME)
            return False
        _cap = cap
        _log.info(
            'Camera opened via ffmpeg AVFoundation (requested=%r, resolved=%r, %dx%d @ %dfps)',
            CAMERA_DEVICE_NAME,
            cap.device_label,
            config.CAMERA_WIDTH,
            config.CAMERA_HEIGHT,
            config.CAMERA_FPS,
        )
        return True

    if CAMERA_INDEX is None:
        _log.warning("Camera open skipped — no camera source configured")
        return False

    cap = cv2.VideoCapture(CAMERA_INDEX)
    if not cap.isOpened():
        cap.release()
        _log.warning("Camera open failed (index=%d)", CAMERA_INDEX)
        return False
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, config.CAMERA_WIDTH)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, config.CAMERA_HEIGHT)
    _cap = cap
    _log.info(
        "Camera opened via OpenCV (index=%d, %dx%d)",
        CAMERA_INDEX, config.CAMERA_WIDTH, config.CAMERA_HEIGHT,
    )
    return True


def _close_camera() -> None:
    global _cap
    if _cap is not None:
        _cap.release()
        _cap = None
        _log.info("Camera closed")


def _mark_camera_offline() -> None:
    global _offline_since
    with _reconnect_lock:
        if _offline_since is None:
            _offline_since = time.monotonic()


def _notify_camera_reconnected_if_needed() -> None:
    global _offline_since
    with _reconnect_lock:
        offline_since = _offline_since
        if offline_since is None:
            return
        _offline_since = None
        callbacks = list(_reconnect_callbacks)

    downtime = max(0.0, time.monotonic() - offline_since)
    _log.info("Camera frame restored after %.1fs offline", downtime)
    for callback in callbacks:
        try:
            callback(downtime)
        except Exception as exc:
            _log.debug("Camera reconnect callback failed: %s", exc)


def _capture_loop() -> None:
    """Daemon thread: reads frames continuously and stores the latest in the shared buffer."""
    global _frame, _last_frame_at

    if not _open_camera():
        _mark_camera_offline()
        _log.warning(
            "Initial camera open failed — will retry every %.1fs",
            config.CAMERA_RECONNECT_INTERVAL_SECS,
        )

    while not _stop_event.is_set():
        if _cap is None or not _cap.isOpened():
            from state import get_state, State
            if get_state() == State.SHUTDOWN:
                break
            _log.info(
                "Attempting camera reconnection (%s)", CAMERA_SELECTION_DESCRIPTION
            )
            _close_camera()
            _stop_event.wait(config.CAMERA_RECONNECT_INTERVAL_SECS)
            if not _open_camera():
                _mark_camera_offline()
            continue

        ret, frame = _cap.read()
        if not ret:
            _log.warning(
                "Camera read failed — treating as disconnect (%s)",
                CAMERA_SELECTION_DESCRIPTION,
            )
            _mark_camera_offline()
            _close_camera()
            continue

        with _frame_lock:
            _frame = frame
            _last_frame_at = time.monotonic()
        _notify_camera_reconnected_if_needed()

    _close_camera()
    _log.info("Camera capture thread stopped")


def _resolve_avfoundation_device_name(name_hint: str) -> str:
    """Resolve a user-friendly macOS camera hint to a concrete AVFoundation device name."""
    device_names = _list_avfoundation_video_devices()
    if not device_names:
        return name_hint

    hint = name_hint.strip().lower()
    exact_matches = [name for name in device_names if name.lower() == hint]
    if exact_matches:
        return exact_matches[0]

    builtin_hints = {"builtin", "built-in", "facetime", "face time", "macbook"}
    if hint in builtin_hints:
        builtin_matches = [
            name for name in device_names
            if "facetime" in name.lower() or "built-in" in name.lower()
        ]
        if len(builtin_matches) == 1:
            return builtin_matches[0]

    substring_matches = [name for name in device_names if hint in name.lower()]
    if len(substring_matches) == 1:
        return substring_matches[0]

    if len(substring_matches) > 1:
        _log.warning(
            "Camera name hint %r matched multiple devices: %s — using literal value",
            name_hint,
            ", ".join(substring_matches),
        )

    return name_hint


def _list_avfoundation_video_devices() -> list[str]:
    """Return macOS video device names reported by ffmpeg, or an empty list on failure."""
    if shutil.which("ffmpeg") is None:
        return []

    try:
        result = subprocess.run(
            ["ffmpeg", "-f", "avfoundation", "-list_devices", "true", "-i", ""],
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            timeout=5.0,
            check=False,
        )
    except (OSError, subprocess.SubprocessError):
        return []

    lines = result.stdout.splitlines()
    device_names: list[str] = []
    in_video_section = False
    for line in lines:
        if "AVFoundation video devices:" in line:
            in_video_section = True
            continue
        if "AVFoundation audio devices:" in line:
            break
        if not in_video_section:
            continue
        match = re.search(r"\[\d+\]\s+(.+)$", line)
        if match:
            device_names.append(match.group(1).strip())
    return device_names
