"""Serialize sounddevice's single global output stream to survive barge-in.

PortAudio/CoreAudio expose ONE process-global output stream through the
``sd.play()`` / ``sd.stop()`` convenience functions. The app legitimately
interrupts playback from another thread — that's the wake-word barge-in feature
(say a wake word to cut Rex off mid-sentence): the speech-queue preempt path and
``interaction._speak_blocking`` call ``sd.stop()`` while the worker thread is
inside ``sd.play()``/``sd.wait()``, then the interrupt-ack ("what?") is replayed
immediately. That cross-thread ``sd.stop()`` followed by an immediate
``sd.play()`` can re-initialize the global stream mid-teardown and HARD-CRASH the
process (observed on macOS as ``Trace/BPT trap: 5``).

``install()`` wraps ``sd.play``/``sd.stop`` ONCE with a shared lock so the two can
never execute concurrently, and leaves a brief settle after a stop so CoreAudio
releases the device before a replay's ``sd.play()`` runs. ``sd.wait()`` is
deliberately NOT wrapped: it blocks for the whole clip and must stay interruptible
by a ``stop()`` on another thread, which is exactly how barge-in works.

This keeps the barge-in feature intact while making the stop→replay sequence
crash-safe. It does not touch PortAudio's realtime audio callback, only the
control-plane start/stop calls, so it can't cause audio dropouts.
"""

import logging
import threading
import time

logger = logging.getLogger(__name__)

# Held only during the brief sd.play()/sd.stop() control calls (and the post-stop
# settle) — never during the blocking sd.wait(). Re-entrant so sounddevice's own
# internal teardown inside play() can't deadlock against us.
_io_lock = threading.RLock()
_install_lock = threading.Lock()
_installed = False

# Per-thread flag: True while THIS thread is inside _guarded_play(). sounddevice's
# play() calls stop() internally before starting each clip; that internal stop must
# NOT incur the post-stop settle (it would pad a gap before every clip and glitch
# playback). Only an explicit barge-in stop on another path should settle. Thread-
# local so a cross-thread barge-in stop (different thread) still settles correctly.
_local = threading.local()

_DEFAULT_STOP_SETTLE_SECS = 0.05


def _stop_settle_secs() -> float:
    try:
        import config
        return max(
            0.0,
            float(getattr(config, "AUDIO_PLAYBACK_STOP_SETTLE_SECS", _DEFAULT_STOP_SETTLE_SECS)),
        )
    except Exception:
        return _DEFAULT_STOP_SETTLE_SECS


def install() -> bool:
    """Idempotently wrap ``sounddevice.play``/``stop`` with the serialization guard.

    Returns True if the guard is active (installed now or already installed),
    False if sounddevice is unavailable. Safe to call repeatedly.
    """
    global _installed
    with _install_lock:
        if _installed:
            return True
        try:
            import sounddevice as sd
        except Exception as exc:
            logger.debug("[sd_guard] sounddevice unavailable; guard not installed: %s", exc)
            return False

        _orig_play = sd.play
        _orig_stop = sd.stop

        def _guarded_play(*args, **kwargs):
            # play() is non-blocking (it starts the stream and returns); hold the
            # lock only for that brief start so a concurrent stop() can't race it.
            with _io_lock:
                _local.in_play = True
                try:
                    result = _orig_play(*args, **kwargs)
                finally:
                    _local.in_play = False
            # Feed exactly what we just started playing to the (optional) software
            # echo canceller as its reference. Skipped entirely unless AEC is enabled
            # (it ships off), so there's zero per-clip overhead in the normal path.
            # Outside the io lock; failures never affect playback.
            try:
                import config as _config
                if getattr(_config, "AEC_SOFTWARE_ENABLED", False):
                    data = args[0] if args else kwargs.get("data")
                    sr = args[1] if len(args) > 1 else kwargs.get("samplerate")
                    if data is not None and sr:
                        from audio import aec
                        aec.push_reference(data, int(sr))
            except Exception:
                pass
            return result

        def _guarded_stop(*args, **kwargs):
            with _io_lock:
                result = _orig_stop(*args, **kwargs)
                # Skip the settle for the stop() that play() issues internally before
                # each clip (same thread, in_play set) — that gap is what caused the
                # startup TTS pauses/glitching. Only an explicit barge-in stop settles.
                if not getattr(_local, "in_play", False):
                    settle = _stop_settle_secs()
                    if settle > 0:
                        # Hold the lock through the settle so a replay's play() waits
                        # for the device to release before re-initializing the stream.
                        time.sleep(settle)
                return result

        sd.play = _guarded_play
        sd.stop = _guarded_stop
        _installed = True
        logger.info("[sd_guard] sounddevice play/stop serialized (settle %.0f ms)", _stop_settle_secs() * 1000)
        return True


def is_installed() -> bool:
    return _installed
