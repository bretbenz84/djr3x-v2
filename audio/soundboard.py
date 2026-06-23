"""
audio/soundboard.py — one-shot MP3 sound-clip player for the gamepad soundboard.

The 8BitDo Pro 2's action buttons (forwarded by the ESP32 firmware as ``button``
events over the motion serial link) can trigger pre-recorded clips from
``assets/audio/clips/``. Clips play on the same output device as TTS, so this
mirrors the TTS playback discipline:

- **no-audio-safe** — skips under ``--noaudio`` / ``AUDIO_OUTPUT_SUPPRESSED``;
- **serialized through the output gate** — a clip never plays over a reply (if Rex
  is speaking the gate is busy and the clip is dropped, not overlapped);
- **mic-suppressed** during + briefly after playback (``echo_cancel``), so Rex
  doesn't hear / react to his own clip (ties into the scene-analyzer guard);
- **non-blocking** — playback runs on a daemon thread so the motion reader thread
  that triggered it is never blocked for the clip's duration.

Decoding reuses ``soundfile`` (libsndfile), the same path TTS uses for its MP3 cache.
"""
from __future__ import annotations

import logging
import threading
from pathlib import Path
from typing import Optional

import numpy as np

import config

_log = logging.getLogger(__name__)
_play_lock = threading.Lock()  # one clip at a time

_AUDIO_EXTS = (".mp3", ".wav", ".ogg", ".flac", ".m4a")


def _clips_dir() -> Path:
    return Path(getattr(config, "SOUNDBOARD_CLIPS_DIR", "assets/audio/clips"))


def resolve_clip(name: str) -> Optional[Path]:
    """Resolve a clip name to a file in the clips dir. Accepts the bare stem
    (``"Air Horn"``) or a filename (``"Air Horn.mp3"``); matches case-insensitively
    and tolerates the configured extension being any supported audio type."""
    name = (name or "").strip()
    if not name:
        return None
    d = _clips_dir()
    has_ext = name.lower().endswith(_AUDIO_EXTS)
    target_name = (name if has_ext else name + ".mp3").lower()
    target_stem = (Path(name).stem if has_ext else name).lower()
    # Scan the directory (never `(dir / requested).exists()` — a case-insensitive
    # filesystem would report True and hand back the REQUESTED casing instead of the
    # real file name). Return the actual file path so it's correct on case-sensitive
    # filesystems too.
    try:
        files = [f for f in d.iterdir() if f.is_file()]
    except OSError:
        return None
    for f in files:  # exact filename, case-insensitive
        if f.name.lower() == target_name:
            return f
    for f in files:  # stem match against any supported extension
        if f.suffix.lower() in _AUDIO_EXTS and f.stem.lower() == target_stem:
            return f
    return None


def list_clips() -> "list[str]":
    """Sorted clip stems available to map to buttons (for tooling / validation)."""
    try:
        return sorted(
            {f.stem for f in _clips_dir().iterdir()
             if f.is_file() and f.suffix.lower() in _AUDIO_EXTS}
        )
    except OSError:
        return []


def _audio_suppressed() -> bool:
    return bool(getattr(config, "AUDIO_OUTPUT_SUPPRESSED", False)) or bool(
        getattr(config, "NO_AUDIO_MODE", False)
    )


def _decode(path: Path):
    try:
        import soundfile as sf

        audio, samplerate = sf.read(str(path), dtype="float32", always_2d=False)
        if getattr(audio, "ndim", 1) > 1:
            audio = audio.mean(axis=1)  # downmix to mono, like tts._read_audio
        return np.asarray(audio, dtype=np.float32), int(samplerate)
    except Exception as exc:
        _log.error("[soundboard] decode failed for %s: %s", path.name, exc)
        return None, 0


def play(name: str, *, async_: bool = True) -> bool:
    """Play clip ``name``. Returns True if a clip was found and (async) started or
    (sync) played to completion; False on no-audio mode, a missing clip, a decode
    error, or a busy output device. Never raises."""
    if _audio_suppressed():
        _log.debug("[soundboard] no-audio mode — skipping clip %r", name)
        return False
    path = resolve_clip(name)
    if path is None:
        _log.warning("[soundboard] clip not found: %r (looked in %s)", name, _clips_dir())
        return False
    if async_:
        threading.Thread(
            target=_play_path, args=(path,), daemon=True, name="soundboard"
        ).start()
        return True
    return _play_path(path)


def _play_path(path: Path) -> bool:
    try:
        import sounddevice as sd
    except ImportError:
        _log.error("[soundboard] sounddevice not installed — cannot play clips")
        return False
    from audio import echo_cancel, output_gate

    audio, samplerate = _decode(path)
    if audio is None or audio.size == 0 or samplerate <= 0:
        return False
    tail = float(getattr(config, "SOUNDBOARD_SUPPRESS_TAIL_SECS", 0.4))
    with _play_lock:
        with output_gate.hold("soundboard") as acquired:
            if not acquired:
                _log.debug("[soundboard] output busy — dropping clip %s", path.stem)
                return False
            try:
                echo_cancel.set_playing(True)
                _log.info("[soundboard] ▶ %s", path.stem)
                sd.play(audio, samplerate, blocksize=2048)
                sd.wait()
                return True
            except Exception as exc:
                _log.error("[soundboard] playback error for %s: %s", path.name, exc)
                return False
            finally:
                echo_cancel.set_playing(False, tail_secs=tail)
