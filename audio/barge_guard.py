"""Last-moment "is the user already talking?" check for proactive speech.

Rex chooses to say some lines on his own — idle banter, idle follow-ups, and the
consciousness greetings/check-ins. Between deciding to speak and the audio
actually playing he spends ~1-2s generating the line and fetching TTS, and the
interaction VAD loop is blocked through that whole window. So a user who starts
answering during that gap isn't noticed, and Rex plays right over them.

This module re-checks the (un-attenuated) rolling mic buffer immediately before
playback so the caller can yield the floor instead. It does NOT try to detect
speech *during* Rex's own playback — that's the half-duplex limit that needs
hardware AEC. It only catches speech that began before/at the moment Rex starts,
which is the common case. See PROACTIVE_SPEECH_YIELD_* in config.

Read-only on the shared audio buffer and safe to call from any thread (VAD model
access is serialized in audio.vad).
"""

import logging
import time
from typing import Optional

import config
from audio import echo_cancel, stream, vad

logger = logging.getLogger(__name__)

# How often to re-sample the mic during the forward-poll window.
_POLL_INTERVAL_SECS = 0.04


def user_speaking_now(
    window_secs: Optional[float] = None,
    min_speech_secs: Optional[float] = None,
    poll_secs: Optional[float] = None,
) -> bool:
    """Return True if the user appears to be (or to start) speaking right now.

    Scans the last ``window_secs`` of the rolling buffer for at least
    ``min_speech_secs`` of speech, and keeps re-checking for up to ``poll_secs``
    so a reply that BEGINS in the same beat the caller is about to speak is still
    caught — not just one already in progress. Returns early the instant speech is
    seen. Returns False (do not yield) while Rex's own playback is suppressing the
    mic, since the buffer would then hold his voice rather than the user's.
    Defaults come from config; ``poll_secs=0`` does a single look-back only.
    """
    window = float(
        window_secs
        if window_secs is not None
        else getattr(config, "PROACTIVE_SPEECH_YIELD_WINDOW_SECS", 0.6)
    )
    min_speech = float(
        min_speech_secs
        if min_speech_secs is not None
        else getattr(config, "PROACTIVE_SPEECH_YIELD_MIN_SPEECH_SECS", 0.1)
    )
    poll = float(
        poll_secs
        if poll_secs is not None
        else getattr(config, "PROACTIVE_SPEECH_YIELD_POLL_SECS", 0.35)
    )
    if window <= 0.0 or min_speech <= 0.0:
        return False

    deadline = time.monotonic() + max(0.0, poll)
    while True:
        try:
            # While Rex is playing (or in the post-playback tail) the buffer holds
            # his own voice — can't trust it to mean "the user is talking".
            if echo_cancel.is_suppressed():
                return False
            audio = stream.get_audio_chunk(window)
            if audio is not None and len(audio) > 0:
                segments = vad.get_speech_segments(audio)
                total_speech = sum(max(0.0, end - start) for start, end in segments)
                if total_speech >= min_speech:
                    return True
        except Exception as exc:
            logger.debug("user_speaking_now check failed: %s", exc)
            return False
        if time.monotonic() >= deadline:
            return False
        time.sleep(_POLL_INTERVAL_SECS)
