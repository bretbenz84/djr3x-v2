"""Serialize libsndfile opens/decodes — its MP3 (mpg123) init is not thread-safe.

Two threads opening MP3 files through ``soundfile`` at the same moment can
hard-crash the process. Observed 2026-08-14 12:48 (``Bus error: 10``,
``EXC_ARM_DA_ALIGN at 0x1``): the TTS worker hit its cache and called
``sf.read()`` on the cached MP3 while another thread was mid-``sf_open`` on a
different MP3 — the crash report shows BOTH threads inside libsndfile's
``mpeg_init``, with the faulting one jumping through a torn pointer (PC 0x1).

The root cause is upstream and acknowledged: libsndfile (1.2.2, bundled in the
``soundfile`` wheel) calls ``mpg123_init()`` on EVERY MPEG open, and its
``src/mpeg_decode.c`` carries a FIXME stating the call "non-conditionally
writes static areas with calculated data" and that libsndfile does not meet
mpg123's call-once threading requirement. So ANY two concurrent MP3 opens race
those global tables — not just the first pair. The app does this legitimately:
TTS cache reads, sound effects, soundboard clips, speech-queue canned lines,
and idle clips all decode on their own threads.

``install()`` wraps ``soundfile.read`` and ``soundfile.SoundFile.__init__``
ONCE with a single re-entrant process lock:

- ``SoundFile.__init__`` is the choke point every open goes through
  (``sf.read``/``sf.write``/``sf.info``/direct construction), and the open is
  where ``mpeg_init`` runs — serializing it removes the crash window.
- ``soundfile.read`` is additionally held for its WHOLE open+decode+close so a
  full MP3 decode can never overlap another open's global-table rewrite. Every
  runtime caller in this codebase reads whole short clips this way; decode cost
  is milliseconds, so the serialization is unobservable in practice.
- The lock is re-entrant because ``sf.read`` constructs a ``SoundFile``
  internally (guarded open inside guarded read, same thread).

Not covered (deliberately): incremental ``SoundFile.read()`` chunk loops after
the open — no runtime code does that today, and post-init decode only READS the
global tables. ``rex_supervisor.py`` (separate, dependency-light process) keeps
its own unguarded fallback reads: it plays at most one rare clip at a time and
prefers ``afplay``.

Sibling of ``sd_guard`` (same shape, same reason): a native audio library with
process-global state, wrapped once at startup, called before the first decode.
"""

import functools
import logging
import threading

logger = logging.getLogger(__name__)

# One lock for open AND whole-file read. RLock: sf.read() constructs SoundFile
# inside the guarded read on the same thread.
_decode_lock = threading.RLock()
_install_lock = threading.Lock()
_installed = False

# Originals, kept as module globals so tests can probe/patch them.
_orig_read = None
_orig_sf_init = None


def install() -> bool:
    """Idempotently wrap ``soundfile.read`` / ``SoundFile.__init__`` with the lock.

    Returns True if the guard is active (installed now or already installed),
    False if soundfile is unavailable. Safe to call repeatedly.
    """
    global _installed, _orig_read, _orig_sf_init
    with _install_lock:
        if _installed:
            return True
        try:
            import soundfile as sf
        except Exception as exc:
            logger.debug("[sndfile_guard] soundfile unavailable; guard not installed: %s", exc)
            return False

        _orig_read = sf.read
        _orig_sf_init = sf.SoundFile.__init__

        @functools.wraps(sf.read)
        def _guarded_read(*args, **kwargs):
            with _decode_lock:
                return _orig_read(*args, **kwargs)

        @functools.wraps(sf.SoundFile.__init__)
        def _guarded_soundfile_init(self, *args, **kwargs):
            with _decode_lock:
                return _orig_sf_init(self, *args, **kwargs)

        sf.read = _guarded_read
        sf.SoundFile.__init__ = _guarded_soundfile_init
        _installed = True
        logger.info(
            "[sndfile_guard] libsndfile open/decode serialized (mpg123 global-init race)"
        )
        return True


def is_installed() -> bool:
    return _installed
