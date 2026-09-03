"""Detect whether a ReSpeaker with onboard hardware AEC is the live audio I/O.

The droid routes BOTH mic capture and Rex's playback through the ReSpeaker —
the Flex XVF3800 Circular-4 since 2026-09-02 (XMOS XVF3800: AEC + 4-mic
beamforming; see docs/respeaker_flex_xvf3800.md), the ReSpeaker Lite before
that (XU316, ~16 dB measured). The chip cancels Rex's voice from the mic. That hardware
cancellation is what makes it safe to shrink the post-TTS "deaf window" so speech
landing as Rex finishes is still captured (see interaction.py boundary handling).

On a dev macOS machine (built-in mic / speakers, AirPods, or any non-ReSpeaker
device) there is NO hardware AEC, so every AEC-dependent change must stay OFF and
the original flush/suppression logic must remain intact. ``is_active()`` is the
single gate for that: it is True only when BOTH the resolved input and output
devices are the ReSpeaker (the name hint below matches both boards).

Override with env ``HARDWARE_AEC=on|off`` (default ``auto`` = device detection).
The result is cached after first resolution; call ``reset_cache()`` in tests.
"""

import logging
import os

import config
from utils.config_loader import AUDIO_DEVICE_INDEX

_log = logging.getLogger(__name__)

_HINT = str(getattr(config, "HARDWARE_AEC_DEVICE_HINT", "respeaker")).strip().lower()
_cached: "bool | None" = None


def _device_name(idx) -> str:
    try:
        import sounddevice as sd
        return str(sd.query_devices(idx).get("name") or "").lower()
    except Exception:
        return ""


def _resolve_output_index() -> "int | None":
    """Resolve the configured playback device the same way main._configure_audio_output_device does."""
    idx = int(getattr(config, "AUDIO_OUTPUT_DEVICE_INDEX", -1))
    if idx >= 0:
        return idx
    name = str(getattr(config, "AUDIO_OUTPUT_DEVICE_NAME", "") or "").strip().lower()
    if not name:
        return None
    try:
        import sounddevice as sd
        for i, dev in enumerate(sd.query_devices()):
            if name in str(dev.get("name", "")).lower() and int(dev.get("max_output_channels", 0)) > 0:
                return i
    except Exception:
        return None
    return None


def _detect() -> bool:
    override = os.getenv("HARDWARE_AEC", "auto").strip().lower()
    if override in {"on", "1", "true", "yes"}:
        return True
    if override in {"off", "0", "false", "no"}:
        return False

    # auto: require the ReSpeaker on BOTH the mic and the speaker path.
    if AUDIO_DEVICE_INDEX is None:
        return False
    if _HINT not in _device_name(AUDIO_DEVICE_INDEX):
        return False
    out_idx = _resolve_output_index()
    if out_idx is None:
        return False
    if _HINT not in _device_name(out_idx):
        return False
    return True


def is_active() -> bool:
    """True only when the ReSpeaker is the live input AND output device.

    Cached after first call. The single gate for every hardware-AEC-dependent
    behavior change — must stay False on non-ReSpeaker (dev) machines.
    """
    global _cached
    if _cached is None:
        _cached = _detect()
        _log.info(
            "[hardware_aec] active=%s (a ReSpeaker on both input and output required)",
            _cached,
        )
    return _cached


def reset_cache() -> None:
    """Force re-detection on the next is_active() call (tests / device changes)."""
    global _cached
    _cached = None
