"""
DJ mode — music request resolution and audio playback.

Call handle_request() to resolve a natural-language request to a TrackInfo.
Call play() to start playback in a background thread.

Playback decodes audio via ffmpeg subprocess and streams raw PCM to sounddevice.
Sources are internet radio streams (PLS → stream URL).
"""

import logging
import os
import re
import shutil
import subprocess
import threading
from collections import namedtuple
from typing import Optional

import numpy as np
import requests
import sounddevice as sd
from rapidfuzz import fuzz

import config
from audio import echo_cancel
from audio import sd_guard
from hardware import leds_head

logger = logging.getLogger(__name__)

TrackInfo = namedtuple("TrackInfo", ["source", "name", "url_or_path", "description"])
# source: "radio"

# ── Module state ──────────────────────────────────────────────────────────────

_stop_event = threading.Event()
_thread: Optional[threading.Thread] = None
_thread_lock = threading.Lock()
_volume: float = 1.0
_VOLUME_STEP = 0.1

_SAMPLE_RATE = 44100
_CHANNELS = 2
_CHUNK_FRAMES = 2048  # ~46 ms per chunk at 44100 Hz


def _ffmpeg_executable() -> str:
    """Return ffmpeg from PATH or common Homebrew locations."""
    resolved = shutil.which("ffmpeg")
    if resolved:
        return resolved
    for candidate in ("/opt/homebrew/bin/ffmpeg", "/usr/local/bin/ffmpeg"):
        if os.path.isfile(candidate) and os.access(candidate, os.X_OK):
            return candidate
    return "ffmpeg"


def _body_beat(name: str) -> None:
    """Trigger a short physical DJ flourish without blocking playback."""
    try:
        from sequences import animations
        animations.play_body_beat(name)
    except Exception as exc:
        logger.debug("[dj] body beat %s skipped: %s", name, exc)

# ── Request resolution ────────────────────────────────────────────────────────

def handle_request(request_text: str) -> Optional[TrackInfo]:
    """
    Resolve a natural-language music request to a TrackInfo by vibe-matching it
    against the radio station vibe tags.

    Returns None if nothing scores above the confidence threshold.
    """
    req = request_text.strip()
    return _vibe_match(req)


def _vibe_match(request_text: str) -> Optional[TrackInfo]:
    """
    Score every radio station vibe tag against the request.
    Returns the highest-scoring TrackInfo above the 50-point threshold, or None.
    """
    req = request_text.lower()
    normalized_req = _normalize_vibe_text(req)
    best_score = 0
    best: Optional[TrackInfo] = None

    for station in config.RADIO_STATIONS:
        for vibe in station["vibes"]:
            score = _station_vibe_score(normalized_req, str(vibe))
            if score > best_score:
                best_score = score
                best = TrackInfo(
                    source="radio",
                    name=station["name"],
                    url_or_path=station["url"],
                    description=(
                        f"Streaming {station['name']} — "
                        + ", ".join(station["vibes"][:3])
                    ),
                )

    if best_score >= 50:
        return best
    return None


def _normalize_vibe_text(text: str) -> str:
    return " ".join(re.sub(r"[^a-z0-9\s-]", " ", (text or "").lower()).split())


def _station_vibe_score(normalized_request: str, vibe: str) -> float:
    """Score station tags conservatively to avoid classic/classical false hits."""
    normalized_vibe = _normalize_vibe_text(vibe)
    if not normalized_request or not normalized_vibe:
        return 0.0

    req_tokens = set(normalized_request.split())
    vibe_tokens = set(normalized_vibe.split())
    if normalized_vibe in normalized_request:
        return 100.0
    if vibe_tokens and vibe_tokens.issubset(req_tokens):
        return 100.0

    score = fuzz.WRatio(normalized_request, normalized_vibe)
    return score if score >= 85 else 0.0


# ── Playback controls ─────────────────────────────────────────────────────────

def play(track_info: TrackInfo) -> None:
    """Start playback of track_info in a background thread. Stops any current playback."""
    global _thread

    stop(beat=False)

    _stop_event.clear()
    with _thread_lock:
        _thread = threading.Thread(
            target=_playback_loop,
            args=(track_info,),
            daemon=True,
            name="dj-playback",
        )
        _thread.start()
    _body_beat("proud_dj_pose")
    try:
        from audio import sound_effects
        sound_effects.play("song_recognized")   # sparkly arpeggio flourish as the track drops
    except Exception:
        pass
    logger.info("[dj] Playing: %s (%s)", track_info.name, track_info.source)


def stop(*, beat: bool = True) -> None:
    """Signal the playback thread to stop and wait for it to exit."""
    with _thread_lock:
        t = _thread
    was_playing = bool(t and t.is_alive() and not _stop_event.is_set())
    _stop_event.set()
    if t and t.is_alive():
        t.join(timeout=3.0)
    echo_cancel.set_playing(False)
    try:
        leds_head.speak_stop()
    except Exception:
        pass
    if beat and was_playing:
        _body_beat("dramatic_visor_peek")


def skip() -> None:
    """Skip the current track (stops playback; caller decides what plays next)."""
    stop()


def set_volume(level: float) -> None:
    """Set playback volume. level is clamped to 0.0–1.0."""
    global _volume
    _volume = max(0.0, min(1.0, float(level)))
    logger.debug("[dj] Volume → %.2f", _volume)


def get_volume() -> float:
    """Return current playback volume."""
    return _volume


def volume_up(step: float = _VOLUME_STEP) -> float:
    """Increase playback volume by step and return the new level."""
    set_volume(_volume + step)
    return _volume


def volume_down(step: float = _VOLUME_STEP) -> float:
    """Decrease playback volume by step and return the new level."""
    set_volume(_volume - step)
    return _volume


def is_playing() -> bool:
    """Return True if the playback thread is running."""
    with _thread_lock:
        t = _thread
    return t is not None and t.is_alive() and not _stop_event.is_set()


# ── Internal playback loop ────────────────────────────────────────────────────

def _resolve_stream_url(pls_url: str) -> Optional[str]:
    """Fetch and parse a .pls file; return the first stream URL found."""
    try:
        resp = requests.get(pls_url, timeout=10)
        resp.raise_for_status()
        for line in resp.text.splitlines():
            stripped = line.strip()
            if not stripped or "=" not in stripped:
                continue
            key, _, val = stripped.partition("=")
            key_lower = key.strip().lower()
            if key_lower.startswith("file") and key_lower[4:].isdigit():
                return val.strip()
    except Exception as exc:
        logger.error("[dj] PLS fetch failed for %s: %s", pls_url, exc)
    return None


def _playback_loop(track_info: TrackInfo) -> None:
    """
    Decode audio with ffmpeg and write PCM chunks to sounddevice.
    Drives mouth LEDs from chunk RMS while playing.
    Runs in a daemon thread; exits cleanly when _stop_event is set.
    """
    echo_cancel.set_playing(True)

    audio_url = _resolve_stream_url(track_info.url_or_path)
    if not audio_url:
        logger.error("[dj] Could not resolve stream URL from %s", track_info.url_or_path)
        echo_cancel.set_playing(False)
        return

    cmd = [
        _ffmpeg_executable(),
        "-hide_banner", "-loglevel", "error",
        "-i", audio_url,
        "-f", "f32le",
        "-ar", str(_SAMPLE_RATE),
        "-ac", str(_CHANNELS),
        "pipe:1",
    ]

    proc: Optional[subprocess.Popen] = None
    stream: Optional["sd.OutputStream"] = None
    try:
        proc = subprocess.Popen(
            cmd, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL
        )
        bytes_per_chunk = _CHUNK_FRAMES * _CHANNELS * 4  # float32 = 4 bytes

        # Open the output stream under the shared device-control lock so this raw
        # OutputStream start can't run concurrently with a TTS sd.play()/sd.stop()
        # on the shared CoreAudio device. That race (e.g. a wake-word barge-in
        # stopping music + Rex's voice at once) silently wedges the mic input
        # callback and freezes the rolling buffer — see audio/sd_guard.py.
        with sd_guard.device_control():
            stream = sd.OutputStream(
                samplerate=_SAMPLE_RATE,
                channels=_CHANNELS,
                dtype="float32",
            )
            stream.start()

        # Steady-state writes run OUTSIDE the lock so a multi-minute song never
        # blocks TTS playback for its full duration.
        while not _stop_event.is_set():
            raw = proc.stdout.read(bytes_per_chunk)
            if not raw:
                break
            # Pad final partial chunk so reshape is always valid
            if len(raw) < bytes_per_chunk:
                raw = raw + b"\x00" * (bytes_per_chunk - len(raw))

            chunk = (
                np.frombuffer(raw, dtype=np.float32)
                .reshape(_CHUNK_FRAMES, _CHANNELS)
            )
            chunk = chunk * _volume
            stream.write(chunk)

            rms = float(np.sqrt(np.mean(chunk ** 2)))
            brightness = int(min(255, rms * config.TTS_LED_BRIGHTNESS_SCALE))
            try:
                leds_head.speak_level(brightness)
            except Exception:
                pass

    except Exception as exc:
        logger.error("[dj] Playback error (%s): %s", track_info.name, exc)
    finally:
        if stream is not None:
            # Close under the same lock (with CoreAudio settle) so the device
            # teardown is serialized against — and releases the device before —
            # any barge-in replay's sd.play().
            with sd_guard.device_control(settle=True):
                try:
                    stream.stop()
                    stream.close()
                except Exception as exc:
                    logger.warning("[dj] Error closing output stream: %s", exc)
        if proc and proc.poll() is None:
            proc.terminate()
            try:
                proc.wait(timeout=2.0)
            except subprocess.TimeoutExpired:
                proc.kill()
        echo_cancel.set_playing(False)
        try:
            leds_head.speak_stop()
        except Exception:
            pass
        logger.info("[dj] Playback finished: %s", track_info.name)
