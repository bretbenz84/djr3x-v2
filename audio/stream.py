"""
Continuous microphone capture via sounddevice.

Opens the mic once and never closes it. Audio is written into a rolling circular
buffer by a non-blocking sounddevice callback. Callers read from the buffer via
get_audio_chunk() or get_full_buffer().

A background watchdog guards against a silently-stalled input callback: on macOS,
another stream's open/close on the shared CoreAudio device (e.g. DJ music
playback) can kill this callback with no error and no PortAudio status flag,
freezing the buffer so every consumer (wake word, VAD, transcription, speaker ID)
reads the same stale audio forever. The watchdog timestamps each callback and
reopens the stream when callbacks stop arriving — see config.AUDIO_STALL_*.

If AUDIO_DEVICE_NAME / AUDIO_DEVICE_INDEX is not set in .env the module
initialises as a no-op and all read functions return empty arrays.
"""

import logging
import math
import os
import threading
import time
from collections import deque

import numpy as np

import config
from utils.config_loader import AUDIO_DEVICE_INDEX, AUDIO_SELECTION_DESCRIPTION

_log = logging.getLogger(__name__)

# Fixed frames per callback invocation. 512 samples at 16 kHz = 32 ms, matching
# Silero VAD's preferred chunk size and keeping the callback very fast.
_BLOCKSIZE = 512

# Number of chunks the deque holds so total capacity == AUDIO_BUFFER_SECONDS.
_MAXLEN: int = math.ceil(config.AUDIO_SAMPLE_RATE * config.AUDIO_BUFFER_SECONDS / _BLOCKSIZE)

_buf: deque = deque(maxlen=_MAXLEN)
_buf_lock = threading.Lock()
_stream = None  # sounddevice.InputStream, or None when disabled
_input_channels: int = 1  # actual device channels; set during start()
# When >= 0, read ONLY this mic channel (the ReSpeaker Lite AEC firmware puts the
# echo-cancelled audio on one channel and the raw reference on another — mixing them
# would re-add the echo). -1 ⇒ mix all channels. Set from config in start().
_aec_channel: int | None = None
# Linear makeup gain applied to every captured block (config.AUDIO_INPUT_GAIN).
# 1.0 ⇒ no change. Set from config in start().
_input_gain: float = 1.0

# ── Stall-watchdog state ──────────────────────────────────────────────────────
# Serializes stream lifecycle (open/close/reopen) so the watchdog and stop()
# can't tear the stream down from under each other. Re-entrant so the same
# thread can nest open inside a guarded section.
_stream_lock = threading.RLock()
# time.monotonic() of the most recent callback; 0.0 == none since (re)open.
_last_callback_at: float = 0.0
# True between start() and stop(): callbacks are expected, so a gap means a stall.
_running: bool = False
_last_reopen_at: float = 0.0
_reopen_count: int = 0
# Consecutive reopen attempts that ended in a WEDGE signature (worker stuck
# inside CoreAudio past its budget, or the lock still held by an earlier stuck
# worker). Distinct from a plain reopen failure (device unplugged, enumerating):
# those can recover in-process, a wedge streak cannot — field 2026-08-25 19:08:
# 11 straight wedged attempts, 64s of deafness before the time-based escalation
# fired. Any non-wedged outcome resets the streak.
_wedged_reopen_streak: int = 0
# time.monotonic() of the last callback BEFORE the current outage, or 0.0 when
# the stream is healthy. Drives the fatal-escalation clock below.
_down_since: float = 0.0
_watchdog_thread: "threading.Thread | None" = None
_watchdog_stop = threading.Event()

# Exit code used when a wedged audio device forces a supervised restart.
_DEAD_MIC_EXIT_CODE = 86


# ── Callback ──────────────────────────────────────────────────────────────────

def _callback(indata, frames, time_info, status):  # noqa: ANN001
    # Stamp first so even a status-flagged callback counts as "alive" for the
    # watchdog (a single global float store is atomic under the GIL — no lock).
    global _last_callback_at
    _last_callback_at = time.monotonic()
    if status:
        _log.warning("sounddevice status: %s", status)
    if _input_channels > 1:
        if _aec_channel is not None and 0 <= _aec_channel < _input_channels:
            # Use the hardware-AEC'd channel verbatim — do NOT mix in the raw channel.
            chunk = indata[:, _aec_channel].copy()
        else:
            # Mix stereo → mono by averaging channels so both capsules contribute.
            chunk = indata.mean(axis=1).copy()
    else:
        chunk = indata[:, 0].copy()
    if _input_gain != 1.0:
        # Makeup gain for low-output mics (e.g. stock ReSpeaker Lite at a distance),
        # hard-clipped to [-1, 1] so a loud/close sound can't wrap or overflow.
        chunk = (chunk * _input_gain).clip(-1.0, 1.0)
    with _buf_lock:
        _buf.append(chunk)


# ── Lifecycle ─────────────────────────────────────────────────────────────────

def _open_stream() -> bool:
    """Open the InputStream and assign module state. Caller must hold _stream_lock.

    Tries the configured channel count first, then the device's reported
    max_input_channels, then mono. macOS PortAudio rejects a request for more
    channels than the active device exposes, so a fixed config that assumes the
    ReSpeaker (2-ch) silently disables the mic whenever a mono device
    (built-in/AirPods) is selected instead. Returns True on success.
    """
    global _stream, _input_channels, _last_callback_at

    requested = getattr(config, "AUDIO_INPUT_CHANNELS", config.AUDIO_CHANNELS)

    import sounddevice as sd

    candidates = [requested]
    try:
        device_info = sd.query_devices(AUDIO_DEVICE_INDEX)
        max_in = int(device_info.get("max_input_channels", 0))
        if max_in and max_in not in candidates:
            candidates.append(max_in)
    except Exception:
        pass
    if 1 not in candidates:
        candidates.append(1)

    last_exc = None
    for ch in candidates:
        try:
            stream = sd.InputStream(
                device=AUDIO_DEVICE_INDEX,
                samplerate=config.AUDIO_SAMPLE_RATE,
                channels=ch,
                dtype="float32",
                blocksize=_BLOCKSIZE,
                callback=_callback,
            )
            stream.start()
        except Exception as exc:
            last_exc = exc
            continue
        _stream = stream
        _input_channels = ch
        # Arm the watchdog grace window: count from open, not from the last
        # callback of a prior (possibly stalled) stream.
        _last_callback_at = time.monotonic()
        if ch != requested:
            _log.warning(
                "Audio device %s does not support %d channels; opened with %d-ch instead.",
                AUDIO_SELECTION_DESCRIPTION, requested, ch,
            )
        _log.info(
            "Audio stream started — %s, %d Hz, %d-ch input → mono, %ds buffer.",
            AUDIO_SELECTION_DESCRIPTION,
            config.AUDIO_SAMPLE_RATE,
            ch,
            config.AUDIO_BUFFER_SECONDS,
        )
        return True

    _log.error("Failed to open audio stream (tried channels %s): %s", candidates, last_exc)
    _stream = None
    return False


def start() -> None:
    """Open the microphone and begin filling the rolling buffer."""
    global _aec_channel, _running, _input_gain, _wedged_reopen_streak

    _wedged_reopen_streak = 0

    ch_cfg = int(getattr(config, "AUDIO_AEC_INPUT_CHANNEL", -1))
    _aec_channel = ch_cfg if ch_cfg >= 0 else None
    if _aec_channel is not None:
        _log.info("Audio input will use AEC channel %d only (no channel mixing).", _aec_channel)

    _input_gain = float(getattr(config, "AUDIO_INPUT_GAIN", 1.0) or 1.0)
    if _input_gain != 1.0:
        _log.info("Audio input makeup gain: %.2fx (config.AUDIO_INPUT_GAIN).", _input_gain)

    if AUDIO_DEVICE_INDEX is None:
        _log.warning(
            "AUDIO_DEVICE_NAME/AUDIO_DEVICE_INDEX not set or not resolved in .env — audio stream disabled. "
            "Wake word, VAD, transcription, and speaker ID will not function."
        )
        return

    with _stream_lock:
        if _stream is not None and _stream.active:
            return
        opened = _open_stream()
        _running = opened

    if _running:
        _start_watchdog()


def is_active() -> bool:
    """Return True when the configured microphone stream is open and running."""
    try:
        return bool(_stream is not None and _stream.active)
    except Exception:
        return False


def stop() -> None:
    """Stop and close the microphone stream.

    Every lock acquire here is BOUNDED: a watchdog reopen wedged inside a
    CoreAudio call holds _stream_lock forever (field 2026-07-31 21:56 — the
    ReSpeaker's USB audio hung, the reopen never returned, and this function's
    unbounded acquire deadlocked the whole shutdown until a force-quit). When
    the lock is wedged there is nothing left to close cleanly anyway — the
    process is exiting and the OS reclaims the device."""
    global _stream, _running

    if _stream_lock.acquire(timeout=5.0):
        try:
            _running = False
        finally:
            _stream_lock.release()
    else:
        _running = False   # benign unlocked write: readers poll it without the lock
        _log.error("Audio stream lock wedged (CoreAudio hang?) — skipping clean close.")
        _stop_watchdog()
        return

    # Stop the watchdog first (without holding _stream_lock — it joins a thread
    # that itself takes the lock during a reopen) so it can't reopen mid-teardown.
    _stop_watchdog()

    if not _stream_lock.acquire(timeout=5.0):
        _log.error("Audio stream lock still wedged — leaving the stream to the OS.")
        return
    try:
        if _stream is None:
            return
        try:
            _stream.stop()
            _stream.close()
        except Exception as exc:
            _log.warning("Error closing audio stream: %s", exc)
        finally:
            _stream = None
            _log.info("Audio stream stopped.")
    finally:
        _stream_lock.release()


# ── Stall watchdog ────────────────────────────────────────────────────────────

def _reopen(reason: str) -> bool:
    """Tear down a stalled stream and open a fresh one. Called by the watchdog.

    The device work runs on a THROWAWAY daemon thread with a bounded join.
    old.stop()/close() and _open_stream() are CoreAudio calls that block forever
    when the USB device wedges, and the watchdog is the ONLY thread that would
    ever retry — so calling them inline killed the retry loop along with the
    attempt. Field 2026-08-18 18:20:46: "reopening (attempt 1)" was the last
    audio log of the session; there was no attempt 2, no outcome line, and Rex
    ran deaf for four minutes (still seeing and moving) until the operator
    force-quit. e2dae47 bounded the SHUTDOWN paths against this same wedge; this
    bounds the RECOVERY path.

    An abandoned worker keeps _stream_lock, so later attempts fail fast on the
    bounded acquire instead of piling up threads behind it.
    """
    global _reopen_count, _last_reopen_at, _wedged_reopen_streak

    if not _running:
        return False

    _reopen_count += 1
    _last_reopen_at = time.monotonic()
    _log.warning(
        "[stream_watchdog] mic input stalled (%s) — reopening (attempt %d).",
        reason, _reopen_count,
    )

    budget = max(1.0, float(getattr(config, "AUDIO_STALL_REOPEN_TIMEOUT_SECS", 5.0)))
    outcome: dict = {}

    def _work() -> None:
        global _stream
        if not _stream_lock.acquire(timeout=budget):
            outcome["lock_wedged"] = True
            return
        try:
            if not _running:
                return
            old = _stream
            _stream = None
            if old is not None:
                try:
                    old.stop()
                    old.close()
                except Exception as exc:
                    _log.warning("[stream_watchdog] error closing stalled stream: %s", exc)

            # Drop the frozen audio so consumers don't keep reading the stale
            # samples the wedged callback left behind.
            with _buf_lock:
                _buf.clear()

            outcome["ok"] = _open_stream()
        except Exception as exc:
            outcome["error"] = exc
        finally:
            _stream_lock.release()

    worker = threading.Thread(
        target=_work, daemon=True, name=f"mic-reopen-{_reopen_count}",
    )
    worker.start()
    worker.join(timeout=budget)

    if worker.is_alive():
        _wedged_reopen_streak += 1
        _log.error(
            "[stream_watchdog] reopen wedged inside CoreAudio (>%.0fs) — abandoning that "
            "thread. The watchdog stays up and keeps retrying.", budget,
        )
        return False
    if outcome.get("lock_wedged"):
        _wedged_reopen_streak += 1
        _log.error(
            "[stream_watchdog] stream lock still held by an earlier wedged reopen — "
            "the device has not come back yet."
        )
        return False
    _wedged_reopen_streak = 0
    if outcome.get("error") is not None:
        _log.error("[stream_watchdog] reopen raised: %s", outcome["error"])
        return False
    if outcome.get("ok"):
        _log.info("[stream_watchdog] mic input reopened.")
        return True
    _log.error("[stream_watchdog] mic reopen failed; will retry.")
    return False


def _escalate_dead_mic(down_secs: float) -> None:
    """No in-process recovery is coming — restart so the device comes back clean.

    A wedged CoreAudio device is external (e2dae47's finding still holds), but
    the wedged handles belong to THIS process, so exiting is what actually frees
    them. rex_supervisor sees the child exit and returns to wake-word listening,
    which reopens the device from scratch.

    Staying up is the worse option: Rex keeps seeing, moving and turning, so he
    LOOKS alive while ignoring everyone in the room. That is exactly the
    2026-08-18 field report — the operator had to kill him by hand.
    """
    _log.error(
        "[stream_watchdog] mic input dead for %.0fs across %d reopen attempts — the "
        "audio device is wedged and will not come back in this process.",
        down_secs, _reopen_count,
    )
    if not bool(getattr(config, "AUDIO_STALL_FATAL_RESTART_ENABLED", True)):
        _log.error(
            "[stream_watchdog] auto-restart disabled (AUDIO_STALL_FATAL_RESTART_ENABLED) "
            "— Rex stays up but is DEAF until restarted by hand."
        )
        return

    grace = max(1.0, float(getattr(config, "AUDIO_STALL_FATAL_EXIT_GRACE_SECS", 20.0)))
    _log.error(
        "[stream_watchdog] requesting shutdown so the supervisor can reopen the device "
        "(hard exit in %.0fs if the clean path hangs).", grace,
    )

    def _exit_backstop() -> None:
        # The graceful path plays a power-down clip and closes this same wedged
        # device, so it can hang too. Give it the window, then leave hard.
        time.sleep(grace)
        _log.error(
            "[stream_watchdog] clean shutdown did not finish in %.0fs — exiting hard.",
            grace,
        )
        logging.shutdown()
        os._exit(_DEAD_MIC_EXIT_CODE)

    threading.Thread(target=_exit_backstop, daemon=True, name="mic-dead-exit").start()

    try:
        import state as state_module
        state_module.set_state(state_module.State.SHUTDOWN)
    except Exception as exc:
        _log.error(
            "[stream_watchdog] could not request a clean shutdown (%s) — exiting now.", exc,
        )
        logging.shutdown()
        os._exit(_DEAD_MIC_EXIT_CODE)


def _watchdog_loop() -> None:
    global _down_since, _wedged_reopen_streak

    interval = max(0.05, float(getattr(config, "AUDIO_STALL_CHECK_INTERVAL_SECS", 0.5)))
    timeout = max(0.2, float(getattr(config, "AUDIO_STALL_TIMEOUT_SECS", 1.5)))
    min_spacing = max(0.0, float(getattr(config, "AUDIO_STALL_REOPEN_MIN_SPACING_SECS", 3.0)))
    fatal_after = float(getattr(config, "AUDIO_STALL_FATAL_SECS", 60.0))
    fatal_streak = int(getattr(config, "AUDIO_STALL_FATAL_WEDGED_REOPENS", 4))

    while not _watchdog_stop.wait(interval):
        if not _running:
            continue
        last = _last_callback_at
        if last <= 0.0:
            continue  # no callback since (re)open yet — still in the grace window
        now = time.monotonic()
        if now - last < timeout:
            if _down_since > 0.0:
                _log.info(
                    "[stream_watchdog] mic input recovered after %.0fs down.",
                    now - _down_since,
                )
                _down_since = 0.0
                _wedged_reopen_streak = 0
            continue  # healthy: callbacks are flowing
        if _down_since <= 0.0:
            _down_since = last
        # A device that has ignored every reopen for this long is not coming back
        # on its own — bail out to the supervisor rather than run on deaf.
        if fatal_after > 0.0 and (now - _down_since) >= fatal_after:
            _escalate_dead_mic(now - _down_since)
            break
        # Fast path: reopens that WEDGE (stick inside CoreAudio, or find the lock
        # still held by a stuck predecessor) are categorically unrecoverable in
        # this process — waiting out the full time budget just extends the
        # deafness (field 2026-08-25 19:08: 11 straight wedges, 64s of nothing
        # before the clock fired). A plain reopen failure — device unplugged,
        # still enumerating — resets the streak and keeps the patient clock.
        if fatal_streak > 0 and _wedged_reopen_streak >= fatal_streak:
            _escalate_dead_mic(now - _down_since)
            break
        if now - _last_reopen_at < min_spacing:
            continue  # reopened recently; give the new stream time to warm
        _reopen(f"no mic callback for {now - last:.1f}s")

    _log.info("[stream_watchdog] stopped.")


def _start_watchdog() -> None:
    global _watchdog_thread

    if not bool(getattr(config, "AUDIO_STALL_WATCHDOG_ENABLED", True)):
        return

    with _stream_lock:
        if _watchdog_thread is not None and _watchdog_thread.is_alive():
            return
        _watchdog_stop.clear()
        _watchdog_thread = threading.Thread(
            target=_watchdog_loop, daemon=True, name="mic-stall-watchdog",
        )
        _watchdog_thread.start()
    _log.info(
        "[stream_watchdog] started (stall timeout %.1fs, check every %.1fs).",
        max(0.2, float(getattr(config, "AUDIO_STALL_TIMEOUT_SECS", 1.5))),
        max(0.05, float(getattr(config, "AUDIO_STALL_CHECK_INTERVAL_SECS", 0.5))),
    )


def _stop_watchdog() -> None:
    global _watchdog_thread

    # Bounded acquires for the same reason as stop(): a reopen wedged in
    # CoreAudio holds _stream_lock, and this must still be able to signal the
    # watchdog and return rather than deadlock the teardown behind it.
    if _stream_lock.acquire(timeout=2.0):
        try:
            t = _watchdog_thread
            if t is None:
                return
            _watchdog_stop.set()
        finally:
            _stream_lock.release()
    else:
        t = _watchdog_thread
        _watchdog_stop.set()
        if t is None:
            return

    t.join(timeout=2.0)
    if t.is_alive():
        _log.warning("[stream_watchdog] thread did not stop cleanly.")

    if _stream_lock.acquire(timeout=2.0):
        try:
            _watchdog_thread = None
        finally:
            _stream_lock.release()


def last_callback_age() -> float:
    """Seconds since the last input callback, or inf if none has fired yet.

    Diagnostic / test hook: a healthy stream returns a value near 0; a stalled
    one grows without bound until the watchdog reopens it.
    """
    last = _last_callback_at
    if last <= 0.0:
        return float("inf")
    return time.monotonic() - last


def mic_down_secs() -> float:
    """Seconds the input has been stalled, or 0.0 when it is healthy.

    Diagnostic hook: unlike last_callback_age() this stays anchored to the START
    of the outage across reopen attempts, so it reflects how long Rex has
    actually been deaf.
    """
    if _down_since <= 0.0:
        return 0.0
    return max(0.0, time.monotonic() - _down_since)


def is_stalled() -> bool:
    """True when the stream should be delivering audio but callbacks have stopped."""
    if not _running:
        return False
    timeout = max(0.2, float(getattr(config, "AUDIO_STALL_TIMEOUT_SECS", 1.5)))
    return last_callback_age() >= timeout


# ── Buffer reads ──────────────────────────────────────────────────────────────

def get_full_buffer() -> np.ndarray:
    """Return a copy of all audio currently in the rolling buffer as a 1-D float32 array."""
    with _buf_lock:
        chunks = list(_buf)
    if not chunks:
        return np.zeros(0, dtype=np.float32)
    return np.concatenate(chunks)


def get_audio_chunk(seconds: float) -> np.ndarray:
    """Return the last `seconds` of audio from the buffer as a 1-D float32 array.

    If the buffer contains less than `seconds` of audio, all available audio is
    returned rather than padding with silence.
    """
    samples_needed = int(seconds * config.AUDIO_SAMPLE_RATE)
    audio = get_full_buffer()
    if len(audio) >= samples_needed:
        return audio[-samples_needed:]
    return audio


def flush() -> None:
    """Discard all audio currently in the rolling buffer.

    Called after TTS playback to prevent Rex's own voice tail from being
    picked up as speech onset on the next listening pass.
    """
    with _buf_lock:
        _buf.clear()
