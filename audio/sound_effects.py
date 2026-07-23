"""
audio/sound_effects.py — short droid chirps/whirs layered under Rex's behavior.

Clips live in ``assets/audio/sound_effects/`` (user-generated MP3s, committed). They
accompany three families of behavior, each with its own enable + cooldown:

  SPEECH  an emotion-matched chirp fired the instant a reaction's TTS starts
          GENERATING. This family plays CONCURRENTLY (see below): the chirp fills the
          ~1 s synthesis gap and the reply's TTS plays normally on top of it — the
          chirp's audible content is over in the first second, so the overlap lands in
          its trailing silence (owner spec). Hooked in speech_queue._worker.
  MOTION  drive-base whirs on real motor commands (turn/move/come/arc), hooked in
          motion_controller so both voice commands and autonomy get them.
  SERVO   servo-whir accents on distinct body gestures (body beats, the wave-back),
          hooked in sequences/animations. Never on face-tracking micro-moves.

THE ONE HARD RULE — effects never gate speech. Two playback disciplines:
  * GATED (motion/servo/head-lift): acquires the shared output gate NON-blocking
    (dropped if busy) and is PREEMPTIBLE — every blocking gate acquirer fires
    output_gate's yield hooks before it waits, and the clip stops within ~50 ms.
  * CONCURRENT (speech emotions, play_for_speech): does NOT hold the gate, so TTS
    never waits on it; it plays its full length and TTS overlaps the trailing silence.
    Fires only when the speaker is currently idle, so it can't truncate an in-progress
    reply, and hands mic-suppression to TTS when TTS takes the speaker.
  Both wrap echo_cancel.set_playing so Rex never transcribes his own chirps.

Variants: registry keys map to LISTS of clip stems; multi-entry keys pick uniformly
at random per play (the "_1"/"_2" files). Decoded audio is cached after first use.
"""

from __future__ import annotations

import logging
import random
import threading
import time
from pathlib import Path
from typing import Optional

import numpy as np

import config

_log = logging.getLogger(__name__)

_AUDIO_EXTS = (".mp3", ".wav", ".ogg", ".flac", ".m4a")

# ── Registry: effect key -> clip stem(s). Stems resolve case-insensitively against
# the directory, so the generator's quirky filenames stay untouched on disk. ──────
_REGISTRY: dict = {
    # Emotions (keys match the TTS emotion vocabulary)
    "happy":        ["Droid_Happy_bouncy"],
    "excited":      ["Droid_Excited"],
    "curious":      ["Droid_Curious"],
    "surprised":    ["Droid_Surprised"],
    "sad":          ["Droid_Sad_slow_descent"],
    "angry":        ["angry_robot"],
    "annoyed":      ["Droid_Annoyedgrumpy"],
    "sleepy":       ["Robot_Sleepy_slow_yawn"],
    "warm":         ["Droid_Affectionate"],
    "sarcastic":    ["Droid_Sassy_smug"],
    # Extra expressive keys (usable by future callers / overrides)
    "confused":     ["Droid_Confused_wobbly"],
    "disappointed": ["Droid_Disappointed"],
    "scared":       ["Droid_Scared_startled"],
    "laughing":     ["Droid_Laughing_rapid_1", "Droid_Laughing_rapid_2"],
    "proud":        ["Droid_Proudtriumphant"],
    "embarrassed":  ["Droid_Embarrassed"],
    "mischievous":  ["Droid_Mischievous"],
    "greeting":     ["Droid_Greeting_whiste"],
    "goodbye":      ["Droid_Goodbye_gentle"],
    "face_recognized": ["droid_Face_recognize"],
    "music":        ["droid_Music-reaction"],
    "song_recognized": ["droid_Song_recognize"],
    # Motion / drivetrain (the user's favorites lead)
    "motion_turn":  ["motion_turning"],
    "motion_move":  ["motion_whir"],
    "motion_spinup": ["Wheel_drive_spin-up", "Low_motorized_hum_1", "Low_motorized_hum_2"],
    "arrived":      ["arrived"],
    "slow_down":    ["slow_down"],
    # Head-lift sweeps (hardware/servos.move_to hook — sustained large travel only)
    "headlift_up":   ["droid_hum_upmotion1", "droid_hum_upmotion2"],
    "headlift_down": ["droid_hum_downmotion"],
    # Servo gesture accents (pool — one is picked at random)
    "servo":        ["Soft_robotic_servo", "Short_hydraulic", "Smooth_gimbal_glide",
                     "Rapid_tiny_servo_chatter"],
    "servo_heavy":  ["Slow_heavy_servo_grind", "Pneumatic_hiss", "Dual-tone_motor_whir",
                     "Quick_ratchet-tick"],
    # System / accents
    "boot":         ["Warm_boot-up_chime_2"],
    "power_down":   ["Power-down_descending"],
    "error":        ["Error_buzz,_short"],
    "confirm":      ["Confirmation_blip"],
    "alert":        ["Alert_ping,_two-tone"],
    "attention":    ["Attention-getting_polite"],
    "scanning":     ["Scanning_sweep,_slow"],
    "thinking":     ["robot_Processing_thinking_1", "robot_Processing_thinking_2"],
    "charger_connected": ["droid_gaining_electric"],
    "charger_disconnected": ["droid_losing_electric"],
    "idle_breath":  ["Soft_idle_robot_breathing"],
}

# Cooldown family per key prefix (speech emotions are everything not listed here).
_MOTION_KEYS = {"motion_turn", "motion_move", "motion_spinup", "arrived", "slow_down"}
_SERVO_KEYS = {"servo", "servo_heavy"}
_HEADLIFT_KEYS = {"headlift_up", "headlift_down"}

_lock = threading.Lock()
_decode_cache: dict = {}          # resolved Path -> (np.ndarray float32 mono, samplerate)
_stem_cache: dict = {}            # lowercase stem -> Path (built lazily per dir scan)
_last_play_at: dict = {}          # family -> monotonic ts of last successful start
_last_key_at: dict = {}           # key -> monotonic ts
_yield_event = threading.Event() # set by output_gate's blocking acquirers via the hook


def _registry() -> dict:
    reg = dict(_REGISTRY)
    reg.update(getattr(config, "SOUND_EFFECTS_REGISTRY_OVERRIDES", {}) or {})
    reg.update(getattr(config, "SOUND_EFFECTS_EMOTION_MAP_OVERRIDES", {}) or {})
    return reg


def _enabled() -> bool:
    if not bool(getattr(config, "SOUND_EFFECTS_ENABLED", True)):
        return False
    if bool(getattr(config, "AUDIO_OUTPUT_SUPPRESSED", False)) or bool(
        getattr(config, "NO_AUDIO_MODE", False)
    ):
        return False
    # Never chirp the dev machine's speakers from the unit-test suite (mirrors
    # rex_db.writes_suppressed): motion/animation tests exercise the real hooks.
    # Tests that want the pipeline opt in by patching this via _test_allow_audio.
    if not _test_allow_audio and _under_test_runner():
        return False
    return True


_test_allow_audio = False   # tests patch True to exercise play() end-to-end (mocked sd)


def _under_test_runner() -> bool:
    import os
    import sys
    if os.environ.get("PYTEST_CURRENT_TEST"):
        return True
    argv0 = (sys.argv[0] if sys.argv else "").lower()
    return "pytest" in argv0 or "unittest" in argv0 or argv0.endswith("py.test")


def _family(key: str) -> str:
    if key in _MOTION_KEYS:
        return "motion"
    if key in _SERVO_KEYS:
        return "servo"
    if key in _HEADLIFT_KEYS:
        return "headlift"
    return "speech"


def _family_allowed(family: str) -> bool:
    flag = {
        "speech": "SOUND_EFFECTS_SPEECH_ENABLED",
        "motion": "SOUND_EFFECTS_MOTION_ENABLED",
        "servo": "SOUND_EFFECTS_SERVO_ENABLED",
        "headlift": "SOUND_EFFECTS_HEADLIFT_ENABLED",
    }[family]
    return bool(getattr(config, flag, True))


def _cooldown(family: str) -> float:
    name = {
        "speech": "SOUND_EFFECTS_SPEECH_COOLDOWN_SECS",
        "motion": "SOUND_EFFECTS_MOTION_COOLDOWN_SECS",
        "servo": "SOUND_EFFECTS_SERVO_COOLDOWN_SECS",
        "headlift": "SOUND_EFFECTS_HEADLIFT_COOLDOWN_SECS",
    }[family]
    default = {"speech": 6.0, "motion": 2.5, "servo": 8.0, "headlift": 5.0}[family]
    try:
        return float(getattr(config, name, default))
    except (TypeError, ValueError):
        return default


def _effects_dir() -> Path:
    return Path(getattr(config, "SOUND_EFFECTS_DIR", "assets/audio/sound_effects"))


def _resolve_stem(stem: str) -> Optional[Path]:
    """Case-insensitive stem -> file path (mirrors soundboard.resolve_clip: scan the
    dir, never trust case-insensitive .exists())."""
    key = stem.strip().lower()
    if not key:
        return None
    cached = _stem_cache.get(key)
    if cached is not None and cached.is_file():
        return cached
    try:
        for f in _effects_dir().iterdir():
            if f.is_file() and f.suffix.lower() in _AUDIO_EXTS:
                _stem_cache[f.stem.lower()] = f
    except OSError:
        return None
    return _stem_cache.get(key)


def _decode(path: Path):
    cached = _decode_cache.get(path)
    if cached is not None:
        return cached
    try:
        import soundfile as sf
        audio, samplerate = sf.read(str(path), dtype="float32", always_2d=False)
        if getattr(audio, "ndim", 1) > 1:
            audio = audio.mean(axis=1)
        out = (np.asarray(audio, dtype=np.float32), int(samplerate))
        _decode_cache[path] = out
        return out
    except Exception as exc:
        _log.warning("[sfx] decode failed for %s: %s", path.name, exc)
        return None, 0


# ── Preemption plumbing (registered with output_gate at import) ──────────────────

def yield_output() -> None:
    """Stop any playing effect ASAP — a blocking audio source wants the speaker."""
    _yield_event.set()


try:
    from audio import output_gate as _og
    _og.register_yield_hook(yield_output)
except Exception:  # circular-import safety in odd tool contexts; wiring is best-effort
    pass


# ── Public API ────────────────────────────────────────────────────────────────────

def play(key: str, *, force: bool = False, concurrent: bool = False) -> bool:
    """Fire effect ``key`` asynchronously. Returns True when a playback thread was
    started (cooldowns/enables/no-audio may drop it silently). Never raises, never
    blocks the caller, and never delays other audio (see module docstring).

    ``concurrent=True`` (used by play_for_speech) plays the clip WITHOUT holding the
    output gate so it can overlap the reply's TTS instead of being preempted by it."""
    try:
        if not _enabled():
            return False
        family = _family(key)
        if not force and not _family_allowed(family):
            return False
        stems = _registry().get(key)
        if not stems:
            return False
        now = time.monotonic()
        with _lock:
            if not force:
                if (now - _last_play_at.get(family, 0.0)) < _cooldown(family):
                    return False
                # Same-key dedup at 2x the family cooldown: the SAME chirp twice in a
                # row reads as a glitch even when the family cooldown has lapsed.
                if (now - _last_key_at.get(key, 0.0)) < 2.0 * _cooldown(family):
                    return False
            _last_play_at[family] = now
            _last_key_at[key] = now
        stem = random.choice(list(stems)) if len(stems) > 1 else stems[0]
        path = _resolve_stem(str(stem))
        if path is None:
            _log.warning("[sfx] clip not found for key %r (stem %r in %s)",
                         key, stem, _effects_dir())
            return False
        threading.Thread(
            target=_play_path, args=(path, key, concurrent), daemon=True,
            name="sound-effects",
        ).start()
        return True
    except Exception as exc:
        _log.debug("[sfx] play(%r) failed: %s", key, exc)
        return False


def play_for_speech(emotion: str, tag: Optional[str] = None) -> bool:
    """The speech-queue hook: fire the emotion's chirp as a reaction's TTS starts
    generating. CONCURRENT — the chirp plays through the ~1 s synthesis gap and TTS
    plays normally on top; the chirp's audible content is over in the first second, so
    the overlap lands in its trailing silence (owner spec). It never holds the output
    gate, so TTS is never delayed; it just won't fire if the speaker is already busy
    (so it can't cut off an in-progress reply). Neutral gets nothing."""
    emotion = str(emotion or "").strip().lower()
    if not emotion or emotion == "neutral":
        return False
    if emotion not in _registry():
        return False
    return play(emotion, concurrent=True)


def _play_path(path: Path, key: str, concurrent: bool = False) -> None:
    try:
        import sounddevice as sd
    except ImportError:
        return
    from audio import echo_cancel, output_gate

    audio, samplerate = _decode(path)
    if audio is None or getattr(audio, "size", 0) == 0 or samplerate <= 0:
        return
    try:
        vol = float(getattr(config, "SOUND_EFFECTS_VOLUME", 0.8))
    except (TypeError, ValueError):
        vol = 0.8
    audio = audio * max(0.0, min(1.0, vol))

    if concurrent:
        _play_concurrent(sd, echo_cancel, output_gate, audio, samplerate, path, key)
    else:
        _play_gated(sd, echo_cancel, output_gate, audio, samplerate, path, key)


def _play_gated(sd, echo_cancel, output_gate, audio, samplerate, path, key) -> None:
    """Serialized, preemptible playback for motion/servo/head-lift accents: acquires the
    output gate (dropped if busy) and yields to any blocking source (TTS) within ~50 ms."""
    # Clear BEFORE acquiring: a hook fired after this point (someone about to block on
    # the gate we may win) must be seen by the wait loop below, not erased.
    _yield_event.clear()
    with output_gate.hold("sound-effects", blocking=False) as acquired:
        if not acquired:
            _log.debug("[sfx] output busy — dropped %s", path.stem)
            return
        try:
            echo_cancel.set_playing(True)
            _log.info("[sfx] ▶ %s (%s)", path.stem, key)
            sd.play(audio, samplerate, blocksize=2048)
            deadline = time.monotonic() + (audio.shape[0] / float(samplerate)) + 0.1
            while time.monotonic() < deadline:
                if _yield_event.wait(timeout=0.05):
                    sd.stop()          # speech wants the speaker — hand it over now
                    _log.debug("[sfx] yielded %s to a blocking source", path.stem)
                    break
        except Exception as exc:
            _log.debug("[sfx] playback error for %s: %s", path.name, exc)
        finally:
            try:
                echo_cancel.set_playing(False, tail_secs=0.25)
            except Exception:
                pass


def _play_concurrent(sd, echo_cancel, output_gate, audio, samplerate, path, key) -> None:
    """Emotion chirp that coexists with the reply's TTS. It does NOT hold the output
    gate (so TTS never waits on it) and is NOT preempted — it plays its full length,
    with the trailing silence absorbing any overlap once TTS audio lands. Fires only
    when the speaker is currently idle, so it can never truncate an in-progress reply."""
    if output_gate.is_busy():
        _log.debug("[sfx] speaker busy — dropping concurrent %s", path.stem)
        return
    duration = audio.shape[0] / float(samplerate)
    try:
        echo_cancel.set_playing(True)          # suppress the mic for the chirp
        _log.info("[sfx] ▶ %s (%s, concurrent)", path.stem, key)
        sd.play(audio, samplerate, blocksize=2048)
        # Hold suppression for the clip's length even if TTS's own playback steals the
        # device stream partway through — TTS is speaking, so the mic must stay muted.
        time.sleep(duration)
    except Exception as exc:
        _log.debug("[sfx] concurrent playback error for %s: %s", path.name, exc)
    finally:
        try:
            # Hand mic suppression to TTS if it took the speaker (it now owns the
            # _playing flag and will release it when the reply ends). Otherwise release
            # with a tail that bridges the chirp->TTS handoff, or restores the mic if
            # this reply turned out to have no spoken audio at all.
            if output_gate.active_source() != "tts":
                echo_cancel.set_playing(False, tail_secs=0.4)
        except Exception:
            pass


def list_effects() -> dict:
    """Key -> resolved file names (diagnostics: which registry entries have files)."""
    out = {}
    for key, stems in sorted(_registry().items()):
        out[key] = [
            (p.name if (p := _resolve_stem(str(s))) is not None else f"MISSING:{s}")
            for s in stems
        ]
    return out


def reset() -> None:
    """Test hook: clear cooldowns + caches."""
    with _lock:
        _last_play_at.clear()
        _last_key_at.clear()
    _stem_cache.clear()
    _yield_event.clear()
