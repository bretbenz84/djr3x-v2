"""
DJ mode — music request resolution and audio playback.

Call scan() at startup to index local MP3 files.
Call handle_request() to resolve a natural-language request to a TrackInfo.
Call play() to start playback in a background thread.

Playback decodes audio via ffmpeg subprocess and streams raw PCM to sounddevice.
This handles both local MP3 files and internet radio streams (PLS → stream URL)
with a single code path.
"""

import logging
import os
import random
import re
import subprocess
import threading
from collections import namedtuple
from typing import Optional

import numpy as np
import requests
import sounddevice as sd
from mutagen.easyid3 import EasyID3
from mutagen import MutagenError
from rapidfuzz import fuzz
from rapidfuzz import process as fuzz_process

import config
from audio import echo_cancel
from hardware import leds_head

logger = logging.getLogger(__name__)

TrackInfo = namedtuple("TrackInfo", ["source", "name", "url_or_path", "description"])
# source: "local" | "radio"

# ── Module state ──────────────────────────────────────────────────────────────

_index: list[dict] = []
_index_lock = threading.Lock()

_stop_event = threading.Event()
_thread: Optional[threading.Thread] = None
_thread_lock = threading.Lock()
_current_track: Optional[TrackInfo] = None
_volume: float = 1.0
_VOLUME_STEP = 0.1

_SAMPLE_RATE = 44100
_CHANNELS = 2
_CHUNK_FRAMES = 2048  # ~46 ms per chunk at 44100 Hz

# ── Music index ───────────────────────────────────────────────────────────────

def scan() -> None:
    """Scan config.MUSIC_DIR for MP3s and build the in-memory index. Call at startup."""
    global _index
    music_dir = config.MUSIC_DIR
    if not os.path.isdir(music_dir):
        logger.warning("[dj] Music directory not found: %s", music_dir)
        return

    found = []
    for root, _, files in os.walk(music_dir):
        for fname in files:
            if not fname.lower().endswith(".mp3"):
                continue
            path = os.path.join(root, fname)
            basename = os.path.splitext(fname)[0]
            try:
                tags = EasyID3(path)
                title  = tags.get("title",  [basename])[0]
                artist = tags.get("artist", [""])[0]
                album  = tags.get("album",  [""])[0]
                genre  = tags.get("genre",  [""])[0]
            except MutagenError:
                title = basename
                artist = album = genre = ""
            found.append({
                "path":   path,
                "title":  title,
                "artist": artist,
                "album":  album,
                "genre":  genre,
            })

    with _index_lock:
        _index = found
    logger.info("[dj] Indexed %d local tracks from %s", len(found), music_dir)


# ── Request resolution ────────────────────────────────────────────────────────

def handle_request(request_text: str) -> Optional[TrackInfo]:
    """
    Resolve a natural-language music request to a TrackInfo.

    Strategies in order:
      1. Title fuzzy match against local index
      2. Artist fuzzy match against local index
      3. Vibe match against radio station vibe tags and local genre tags

    Returns None if nothing scores above the confidence threshold.
    """
    with _index_lock:
        snapshot = list(_index)

    req = request_text.strip()

    # 1. Title fuzzy match
    if snapshot:
        titles = [t["title"] for t in snapshot]
        hit = fuzz_process.extractOne(req, titles, scorer=fuzz.WRatio, score_cutoff=60)
        if hit:
            _, _, idx = hit
            track = snapshot[idx]
            return TrackInfo(
                source="local",
                name=track["title"],
                url_or_path=track["path"],
                description=(
                    f"{track['title']} by {track['artist']}"
                    if track["artist"] else track["title"]
                ),
            )

    # 2. Artist fuzzy match
    if snapshot:
        artist_names = [t["artist"] for t in snapshot if t["artist"]]
        if artist_names:
            hit = fuzz_process.extractOne(
                req, artist_names, scorer=fuzz.WRatio, score_cutoff=60
            )
            if hit:
                matched_artist = hit[0]
                candidates = [t for t in snapshot if t["artist"] == matched_artist]
                track = random.choice(candidates)
                return TrackInfo(
                    source="local",
                    name=track["title"],
                    url_or_path=track["path"],
                    description=f"{track['title']} by {track['artist']}",
                )

    # 3. Vibe match
    return _vibe_match(req, snapshot)


def _vibe_match(request_text: str, snapshot: list[dict]) -> Optional[TrackInfo]:
    """
    Score every radio station vibe tag and local genre tag against the request.
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

    for track in snapshot:
        if not track["genre"]:
            continue
        score = fuzz.partial_ratio(req, track["genre"].lower())
        if score > best_score:
            best_score = score
            best = TrackInfo(
                source="local",
                name=track["title"],
                url_or_path=track["path"],
                description=(
                    f"{track['title']} by {track['artist']}"
                    if track["artist"] else track["title"]
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
    global _thread, _current_track

    stop()

    _stop_event.clear()
    with _thread_lock:
        _current_track = track_info
        _thread = threading.Thread(
            target=_playback_loop,
            args=(track_info,),
            daemon=True,
            name="dj-playback",
        )
        _thread.start()
    logger.info("[dj] Playing: %s (%s)", track_info.name, track_info.source)


def stop() -> None:
    """Signal the playback thread to stop and wait for it to exit."""
    global _current_track
    _stop_event.set()
    with _thread_lock:
        t = _thread
    if t and t.is_alive():
        t.join(timeout=3.0)
    with _thread_lock:
        _current_track = None
    echo_cancel.set_playing(False)
    try:
        leds_head.speak_stop()
    except Exception:
        pass


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


def play_by_vibe(vibe: str) -> Optional[TrackInfo]:
    """Resolve a vibe request and start playback when a match is found."""
    track = handle_request(vibe)
    if track is None:
        logger.info("[dj] No match found for vibe request: %r", vibe)
        return None
    play(track)
    return track


def is_playing() -> bool:
    """Return True if the playback thread is running."""
    with _thread_lock:
        t = _thread
    return t is not None and t.is_alive() and not _stop_event.is_set()


def now_playing() -> Optional[TrackInfo]:
    """Return the currently playing TrackInfo, or None if not playing."""
    if not is_playing():
        return None
    with _thread_lock:
        return _current_track


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

    if track_info.source == "radio":
        audio_url = _resolve_stream_url(track_info.url_or_path)
        if not audio_url:
            logger.error("[dj] Could not resolve stream URL from %s", track_info.url_or_path)
            echo_cancel.set_playing(False)
            return
    else:
        audio_url = track_info.url_or_path

    cmd = [
        "ffmpeg",
        "-hide_banner", "-loglevel", "error",
        "-i", audio_url,
        "-f", "f32le",
        "-ar", str(_SAMPLE_RATE),
        "-ac", str(_CHANNELS),
        "pipe:1",
    ]

    proc: Optional[subprocess.Popen] = None
    try:
        proc = subprocess.Popen(
            cmd, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL
        )
        bytes_per_chunk = _CHUNK_FRAMES * _CHANNELS * 4  # float32 = 4 bytes

        with sd.OutputStream(
            samplerate=_SAMPLE_RATE,
            channels=_CHANNELS,
            dtype="float32",
        ) as stream:
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
