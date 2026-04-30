"""
Auditory scene analysis — ambient level, music, laughter, and applause detection.

Runs as a background daemon thread. Reads from the shared audio stream buffer,
classifies each analysis window using energy and spectral heuristics, and writes
results into world_state.audio_scene. All thresholds are configurable in config.py.

Detection approach:
  Ambient level  — RMS of the full window, classified against two RMS thresholds.
  Music          — FFT band energy across bass/mid/treble; music has energy in
                   multiple bands simultaneously while speech concentrates in mid.
  Laughter       — burst-pattern heuristic: high RMS variance across short
                   sub-windows indicates rhythmic energy spikes characteristic
                   of laughter, combined with a minimum mean energy gate.
  Applause       — sustained broadband noise: high spectral flatness (geometric
                   mean / arithmetic mean of spectrum) plus minimum RMS.

All detectors degrade gracefully on short or empty audio arrays.
"""

import logging
import threading
import time
from datetime import datetime, timezone

import numpy as np

import config
from audio import stream, output_gate, speech_queue
from world_state import world_state

logger = logging.getLogger(__name__)

_stop_event = threading.Event()
_thread: threading.Thread | None = None


# ── Lifecycle ─────────────────────────────────────────────────────────────────

def start() -> None:
    """Start the background analysis thread. Safe to call multiple times."""
    global _thread, _stop_event
    if _thread is not None and _thread.is_alive():
        return
    _stop_event = threading.Event()
    _thread = threading.Thread(target=_analysis_loop, name="audio-scene", daemon=True)
    _thread.start()
    logger.info("Auditory scene analysis started (interval=%.1fs).", config.SCENE_ANALYSIS_INTERVAL_SECS)


def stop() -> None:
    """Signal the analysis thread to stop and wait for it to exit."""
    global _thread
    _stop_event.set()
    if _thread is not None:
        _thread.join(timeout=5.0)
        _thread = None
    logger.info("Auditory scene analysis stopped.")


# ── Analysis loop ─────────────────────────────────────────────────────────────

def _analysis_loop() -> None:
    # _stop_event.wait(timeout) returns True when the event fires (stop requested),
    # False when it times out — so the loop body runs on each timeout.
    while not _stop_event.wait(timeout=config.SCENE_ANALYSIS_INTERVAL_SECS):
        try:
            # Skip while the robot is playing its own audio, and for one window
            # afterward — speaker bleed into the mic produces rhythmic bursts
            # that trip _detect_laughter, and the analysis buffer still holds
            # the playback tail until it's overwritten. seconds_since_release()
            # spans module boundaries, so this works even on the first cycle
            # after startup clips that played before this loop began.
            if (
                speech_queue.is_speaking()
                or output_gate.seconds_since_release() < config.SCENE_ANALYSIS_WINDOW_SECS
            ):
                continue
            audio = stream.get_audio_chunk(config.SCENE_ANALYSIS_WINDOW_SECS)
            _analyze_cycle(audio)
        except Exception as exc:
            logger.error("Scene analysis cycle error: %s", exc)


def _analyze_cycle(audio: np.ndarray) -> None:
    ambient   = _classify_ambient(audio)
    music     = _detect_music(audio)
    laughter  = _detect_laughter(audio)
    applause  = _detect_applause(audio)
    chatter   = _detect_group_chatter(audio, music=music, laughter=laughter, applause=applause)
    now_ts    = time.time()

    scene = world_state.get("audio_scene")
    scene["ambient_level"]      = ambient
    scene["music_detected"]     = music
    scene["laughter_detected"]  = laughter
    scene["applause_detected"]  = applause
    scene["group_chatter_detected"] = chatter
    if chatter:
        hold = float(getattr(config, "GROUP_CHATTER_HOLD_SECS", 6.0))
        scene["group_chatter_until"] = now_ts + max(0.0, hold)
        scene["group_chatter_reason"] = "sustained_speech_density"
    elif scene.get("group_chatter_until") and now_ts > float(scene["group_chatter_until"]):
        scene["group_chatter_until"] = None
        scene["group_chatter_reason"] = None
    scene["last_updated"]       = datetime.now(timezone.utc).isoformat()

    if laughter:
        scene["last_sound_event"] = "laughter"
    elif applause:
        scene["last_sound_event"] = "applause"

    world_state.update("audio_scene", scene)


# ── Detectors ─────────────────────────────────────────────────────────────────

def _classify_ambient(audio: np.ndarray) -> str:
    if len(audio) == 0:
        return "quiet"
    rms = float(np.sqrt(np.mean(audio.astype(np.float32) ** 2)))
    if rms < config.SCENE_AMBIENT_QUIET_RMS:
        return "quiet"
    if rms > config.SCENE_AMBIENT_LOUD_RMS:
        return "loud"
    return "moderate"


def _detect_music(audio: np.ndarray) -> bool:
    """True if energy is present in at least SCENE_MUSIC_ACTIVE_BANDS_MIN frequency bands."""
    sr = config.AUDIO_SAMPLE_RATE
    if len(audio) < sr // 2:
        return False

    # Use the last second of audio for a clean 1 Hz / bin resolution.
    window = audio[-sr:].astype(np.float32)
    # Normalise by window length so magnitudes are comparable to time-domain amplitude.
    spectrum = np.abs(np.fft.rfft(window)) / len(window)
    freqs = np.fft.rfftfreq(len(window), d=1.0 / sr)

    def _band_energy(lo: float, hi: float) -> float:
        mask = (freqs >= lo) & (freqs < hi)
        return float(np.mean(spectrum[mask] ** 2)) if mask.any() else 0.0

    bass   = _band_energy(60,   300)
    mid    = _band_energy(300,  3000)
    treble = _band_energy(3000, 8000)

    active = sum(
        e >= config.SCENE_MUSIC_BAND_ENERGY_MIN
        for e in (bass, mid, treble)
    )
    return active >= config.SCENE_MUSIC_ACTIVE_BANDS_MIN


def _detect_laughter(audio: np.ndarray) -> bool:
    """True when there are rhythmic energy bursts consistent with laughter.

    Divides the window into 50 ms sub-chunks and checks for high variance in
    per-chunk RMS (burst pattern) combined with sufficient mean energy.
    """
    sr = config.AUDIO_SAMPLE_RATE
    chunk_len = sr // 20  # 50 ms
    if len(audio) < chunk_len * 5:
        return False

    window = audio[-int(sr * 1.5):].astype(np.float32)
    n_chunks = len(window) // chunk_len
    rms_values = np.array([
        np.sqrt(np.mean(window[i * chunk_len:(i + 1) * chunk_len] ** 2))
        for i in range(n_chunks)
    ])

    return (
        float(np.mean(rms_values)) >= config.SCENE_LAUGHTER_MEAN_RMS_MIN
        and float(np.var(rms_values)) >= config.SCENE_LAUGHTER_BURST_VARIANCE_MIN
    )


def _detect_applause(audio: np.ndarray) -> bool:
    """True when audio has sustained broadband noise characteristic of applause.

    Uses spectral flatness (geometric mean / arithmetic mean of spectrum magnitudes).
    A flat spectrum (SFM → 1.0) indicates broadband noise; tonal or sparse signals
    score much lower.
    """
    sr = config.AUDIO_SAMPLE_RATE
    if len(audio) < sr // 2:
        return False

    window = audio[-int(sr * 1.5):].astype(np.float32)
    rms = float(np.sqrt(np.mean(window ** 2)))
    if rms < config.SCENE_APPLAUSE_RMS_MIN:
        return False

    spectrum = np.abs(np.fft.rfft(window)) + 1e-10
    flatness = float(np.exp(np.mean(np.log(spectrum))) / np.mean(spectrum))
    return flatness >= config.SCENE_APPLAUSE_SPECTRAL_FLATNESS_MIN


def _detect_group_chatter(
    audio: np.ndarray,
    *,
    music: bool = False,
    laughter: bool = False,
    applause: bool = False,
) -> bool:
    """
    True when recent audio looks like sustained background conversation.

    This is intentionally conservative and non-identifying: it only asks whether
    speech-like energy has been nearly continuous for a few seconds, with enough
    short on/off changes to resemble back-and-forth banter instead of one clear
    addressed utterance.
    """
    if music or applause:
        return False
    if len(audio) == 0:
        return False

    sr = config.AUDIO_SAMPLE_RATE
    min_secs = float(getattr(config, "GROUP_CHATTER_MIN_WINDOW_SECS", 3.0))
    if len(audio) < int(sr * min_secs):
        return False

    window_secs = float(getattr(config, "GROUP_CHATTER_AUDIO_WINDOW_SECS", 4.0))
    window = audio[-int(sr * window_secs):].astype(np.float32)
    chunk_len = max(1, int(sr * float(getattr(config, "GROUP_CHATTER_CHUNK_SECS", 0.08))))
    n_chunks = len(window) // chunk_len
    if n_chunks < 8:
        return False

    rms_values = np.array([
        np.sqrt(np.mean(window[i * chunk_len:(i + 1) * chunk_len] ** 2))
        for i in range(n_chunks)
    ])
    active_floor = float(getattr(config, "GROUP_CHATTER_ACTIVE_RMS_MIN", 0.014))
    active = rms_values >= active_floor
    coverage = float(np.mean(active))
    transitions = int(np.count_nonzero(active[1:] != active[:-1])) if len(active) > 1 else 0

    if laughter and coverage < 0.85:
        return False

    return (
        coverage >= float(getattr(config, "GROUP_CHATTER_MIN_SPEECH_COVERAGE", 0.58))
        and transitions >= int(getattr(config, "GROUP_CHATTER_MIN_ENERGY_TRANSITIONS", 3))
    )
