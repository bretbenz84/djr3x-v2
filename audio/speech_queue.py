"""
Global speech queue — singleton FIFO with priority levels and a background worker.

A single worker thread pulls items from a priority heap and calls tts.speak()
(or plays audio files) sequentially — only one item plays at a time.

Priority levels (higher plays sooner):
    0  background  idle thoughts, presence reactions from consciousness
    1  normal      interaction responses, curiosity questions
    2  urgent      wake acknowledgments, interruptions

When an item is enqueued at priority P:
  - All *waiting* items with priority < P are dropped immediately and their
    done events are set so any blocked caller unblocks.
  - If the worker is *currently playing* an item with priority < P, sd.stop()
    is called to preempt it, giving the new item the next play slot.

clear_below_priority(n) can be called by callers to flush lower-priority
items from the queue, e.g. when an interrupt is being processed.
"""

import heapq
import logging
import threading
from pathlib import Path
from typing import Optional

logger = logging.getLogger(__name__)


# ── Queue item ─────────────────────────────────────────────────────────────────

class _Item:
    __slots__ = (
        "neg_priority", "seq", "text", "emotion", "audio_path",
        "done", "tag", "pre_beat_ms", "post_beat_ms", "voice_settings",
    )

    def __init__(
        self,
        priority: int,
        seq: int,
        text: Optional[str],
        emotion: str,
        audio_path: Optional[str],
        done: threading.Event,
        tag: Optional[str] = None,
        pre_beat_ms: int = 0,
        post_beat_ms: int = 0,
        voice_settings: Optional[dict] = None,
    ) -> None:
        self.neg_priority = -priority
        self.seq = seq
        self.text = text
        self.emotion = emotion
        self.audio_path = audio_path
        self.done = done
        self.tag = tag
        self.pre_beat_ms = pre_beat_ms
        self.post_beat_ms = post_beat_ms
        self.voice_settings = voice_settings

    def __lt__(self, other: "_Item") -> bool:
        if self.neg_priority != other.neg_priority:
            return self.neg_priority < other.neg_priority
        return self.seq < other.seq

    @property
    def priority(self) -> int:
        return -self.neg_priority


# ── Queue implementation ───────────────────────────────────────────────────────

class _SpeechQueue:
    def __init__(self) -> None:
        self._heap: list[_Item] = []
        self._lock = threading.Lock()
        self._not_empty = threading.Condition(self._lock)
        self._seq = 0
        self._speaking = False
        self._current_priority: int = -1
        self._current_audio_path: Optional[str] = None
        self._startup_chime_queued: bool = False

        threading.Thread(
            target=self._worker, daemon=True, name="speech-queue-worker"
        ).start()

    # ── Public API ─────────────────────────────────────────────────────────────

    def enqueue(
        self,
        text: str,
        emotion: str = "neutral",
        priority: int = 0,
        tag: Optional[str] = None,
        pre_beat_ms: int = 0,
        post_beat_ms: int = 0,
        voice_settings: Optional[dict] = None,
    ) -> threading.Event:
        """Enqueue text for TTS. Returns an Event set when playback finishes.

        If tag is given, any waiting items with the same tag are dropped first —
        useful for coalescing stale presence/idle reactions.

        pre_beat_ms / post_beat_ms add a silent pause before / after speaking
        (worker holds the queue open during the beat so nothing else cuts in).

        voice_settings (optional dict of stability / style / similarity_boost /
        use_speaker_boost) overrides ElevenLabs voice parameters for this item.
        Cached separately from the default-voice take.
        """
        return self._add(
            text, emotion, None, priority, tag,
            pre_beat_ms, post_beat_ms, voice_settings,
        )

    def enqueue_audio_file(
        self,
        path: str,
        priority: int = 0,
        tag: Optional[str] = None,
        pre_beat_ms: int = 0,
        post_beat_ms: int = 0,
    ) -> threading.Event:
        """Enqueue an audio file for direct playback. Returns an Event set when done."""
        return self._add(None, "neutral", path, priority, tag, pre_beat_ms, post_beat_ms)

    def drop_by_tag(self, tag: str) -> int:
        """Drop all *waiting* items whose tag matches. Returns count dropped."""
        dropped = 0
        with self._not_empty:
            keep = []
            for item in self._heap:
                if item.tag == tag:
                    item.done.set()
                    dropped += 1
                else:
                    keep.append(item)
            if dropped:
                self._heap = keep
                heapq.heapify(self._heap)
        return dropped

    def has_waiting_with_tag(self, tag: str) -> bool:
        """True if any waiting (not-yet-playing) item has this tag."""
        with self._lock:
            return any(item.tag == tag for item in self._heap)

    def clear_below_priority(self, n: int) -> None:
        """Drop all *waiting* items with priority < n and set their done events."""
        with self._not_empty:
            keep = []
            for item in self._heap:
                if item.priority < n:
                    item.done.set()
                else:
                    keep.append(item)
            if len(keep) != len(self._heap):
                self._heap = keep
                heapq.heapify(self._heap)

    def is_speaking(self) -> bool:
        """True while the worker is actively playing audio."""
        with self._lock:
            return self._speaking

    def current_audio_path(self) -> Optional[str]:
        """Return the direct audio file currently playing, if any."""
        with self._lock:
            return self._current_audio_path

    # ── Internal ───────────────────────────────────────────────────────────────

    def _add(
        self,
        text: Optional[str],
        emotion: str,
        audio_path: Optional[str],
        priority: int,
        tag: Optional[str] = None,
        pre_beat_ms: int = 0,
        post_beat_ms: int = 0,
        voice_settings: Optional[dict] = None,
    ) -> threading.Event:
        done = threading.Event()
        should_preempt = False

        with self._not_empty:
            # Drop all waiting items of strictly lower priority, plus any
            # waiting items with the same tag (coalesce stale reactions).
            keep = []
            for item in self._heap:
                if item.priority < priority or (tag is not None and item.tag == tag):
                    item.done.set()
                else:
                    keep.append(item)
            if len(keep) != len(self._heap):
                self._heap = keep
                heapq.heapify(self._heap)

            # Preempt current playback if it has lower priority
            if self._speaking and self._current_priority < priority:
                should_preempt = True

            if text and not self._startup_chime_queued:
                self._maybe_add_startup_chime_locked(priority)

            seq = self._seq
            self._seq += 1
            heapq.heappush(
                self._heap,
                _Item(priority, seq, text, emotion, audio_path, done, tag,
                      pre_beat_ms, post_beat_ms, voice_settings),
            )
            self._not_empty.notify()

        if should_preempt:
            try:
                import sounddevice as sd
                from audio import echo_cancel
                echo_cancel.request_cancel()
                sd.stop()
            except Exception:
                pass

        return done

    def _maybe_add_startup_chime_locked(self, priority: int) -> None:
        self._startup_chime_queued = True
        try:
            import config
            try:
                from features import games as games_mod
                if games_mod.is_active():
                    logger.debug("speech_queue: first listening chime skipped during active game")
                    return
            except Exception:
                pass

            if not bool(getattr(config, "PLAY_LISTENING_CHIME", True)):
                return
            path = Path(getattr(config, "LISTENING_CHIME_FILE", "") or "")
            if not path.is_absolute():
                path = Path(__file__).resolve().parent.parent / path
            if not path.exists():
                logger.warning("speech_queue: startup listening chime missing: %s", path)
                return
            chime_done = threading.Event()
            seq = self._seq
            self._seq += 1
            heapq.heappush(
                self._heap,
                _Item(
                    priority,
                    seq,
                    None,
                    "neutral",
                    str(path),
                    chime_done,
                    "system:first_listening_chime",
                ),
            )
            logger.info("speech_queue: queued first listening chime before speech: %s", path)
        except Exception as exc:
            logger.debug("speech_queue: first listening chime skipped: %s", exc)

    def reset_startup_chime_for_tests(self) -> None:
        with self._lock:
            self._startup_chime_queued = False

    def _worker(self) -> None:
        while True:
            with self._not_empty:
                while not self._heap:
                    self._not_empty.wait()
                item = heapq.heappop(self._heap)

            with self._lock:
                self._speaking = True
                self._current_priority = item.priority
                self._current_audio_path = item.audio_path

            try:
                try:
                    from awareness.situation import assessor as _sit
                    _sit.set_rex_speaking(True)
                except Exception:
                    pass

                if item.pre_beat_ms > 0:
                    import time as _t
                    _t.sleep(item.pre_beat_ms / 1000.0)

                if item.audio_path:
                    self._play_file(item.audio_path)
                elif item.text:
                    from audio import tts
                    tts.speak(item.text, item.emotion, voice_settings=item.voice_settings)

                if item.post_beat_ms > 0:
                    import time as _t
                    _t.sleep(item.post_beat_ms / 1000.0)
            except Exception as exc:
                logger.error("speech_queue worker error: %s", exc)
            finally:
                try:
                    from awareness.situation import assessor as _sit
                    _sit.set_rex_speaking(False)
                except Exception:
                    pass
                for _cb in _on_item_done_callbacks:
                    try:
                        _cb(item)
                    except TypeError:
                        # Older callbacks accepted no item context.
                        try:
                            _cb()
                        except Exception:
                            pass
                    except Exception:
                        pass
                with self._lock:
                    self._speaking = False
                    self._current_priority = -1
                    self._current_audio_path = None
                item.done.set()

    def _play_file(self, path: str) -> None:
        try:
            import math
            import numpy as np
            import sounddevice as sd
            import soundfile as sf

            from audio import echo_cancel, output_gate
            import config

            audio, samplerate = sf.read(str(path), dtype="float32", always_2d=False)
            if audio.ndim > 1:
                audio = audio.mean(axis=1)
            audio = np.nan_to_num(audio, nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32)

            path_obj = Path(str(path))
            is_jeopardy = path_obj.parent.name == "jeopardy"
            if is_jeopardy:
                target_sr = int(getattr(config, "JEOPARDY_AUDIO_OUTPUT_SAMPLE_RATE", 44100) or 0)
                if target_sr > 0 and samplerate != target_sr and audio.size:
                    try:
                        from scipy.signal import resample_poly
                        common = math.gcd(int(samplerate), int(target_sr))
                        audio = resample_poly(
                            audio,
                            target_sr // common,
                            int(samplerate) // common,
                        ).astype(np.float32)
                        samplerate = target_sr
                    except Exception as exc:
                        logger.debug("speech_queue: jeopardy resample skipped: %s", exc)

                if path_obj.name == "jeopardy-theme.mp3":
                    max_secs = float(getattr(config, "JEOPARDY_THEME_MAX_SECS", 0.0) or 0.0)
                    if max_secs > 0:
                        audio = audio[: int(max_secs * samplerate)]

                music_files = {
                    "jeopardy-intro.mp3",
                    "jeopardy-theme.mp3",
                    "jeopardy-final-jeopardy-thinking-music.mp3",
                    "jeopardy-outro-no-talking.mp3",
                    "jeopardy-daily-double.mp3",
                }
                gain = (
                    float(getattr(config, "JEOPARDY_AUDIO_MUSIC_GAIN", 0.35))
                    if path_obj.name in music_files
                    else float(getattr(config, "JEOPARDY_AUDIO_STINGER_GAIN", 0.75))
                )
                audio = audio * gain

                # Prevent hard clicks at clip boundaries and leave headroom for
                # small speakers that distort well before digital full scale.
                fade_samples = min(int(0.015 * samplerate), max(0, audio.size // 2))
                if fade_samples > 1:
                    fade = np.linspace(0.0, 1.0, fade_samples, dtype=np.float32)
                    audio[:fade_samples] *= fade
                    audio[-fade_samples:] *= fade[::-1]

            peak = float(np.max(np.abs(audio))) if audio.size else 0.0
            if peak > 0.85:
                audio = audio * (0.85 / peak)

            with output_gate.hold("speech-queue-file") as acquired:
                if not acquired:
                    logger.debug("speech_queue: playback skipped — output gate busy")
                    return
                try:
                    echo_cancel.set_playing(True)
                    sd.play(audio, samplerate, blocksize=2048)
                    sd.wait()
                finally:
                    echo_cancel.set_playing(False)
        except Exception as exc:
            logger.error("speech_queue: failed to play file %s: %s", path, exc)


# ── Post-item-done hooks ───────────────────────────────────────────────────────

_on_item_done_callbacks: list = []


def register_on_item_done(fn) -> None:
    """Register a callback invoked after each queue item finishes playing.

    Called from the worker thread before the item's done-event is set, so any
    post-TTS deaf windows are armed before a waiting caller (e.g. _speak_blocking)
    unblocks and the interaction loop resumes listening.
    """
    if fn not in _on_item_done_callbacks:
        _on_item_done_callbacks.append(fn)


# ── Module-level singleton + thin wrappers ─────────────────────────────────────

_queue = _SpeechQueue()


def enqueue(
    text: str,
    emotion: str = "neutral",
    priority: int = 0,
    tag: Optional[str] = None,
    pre_beat_ms: int = 0,
    post_beat_ms: int = 0,
    voice_settings: Optional[dict] = None,
) -> threading.Event:
    """Enqueue text for TTS speech. Returns an Event set when playback finishes."""
    return _queue.enqueue(
        text, emotion, priority, tag, pre_beat_ms, post_beat_ms, voice_settings,
    )


def enqueue_audio_file(
    path: str,
    priority: int = 0,
    tag: Optional[str] = None,
    pre_beat_ms: int = 0,
    post_beat_ms: int = 0,
) -> threading.Event:
    """Enqueue an audio file for playback. Returns an Event set when done."""
    return _queue.enqueue_audio_file(path, priority, tag, pre_beat_ms, post_beat_ms)


def reset_startup_chime_for_tests() -> None:
    _queue.reset_startup_chime_for_tests()


def clear_below_priority(n: int) -> None:
    """Drop all waiting queue items with priority < n."""
    _queue.clear_below_priority(n)


def drop_by_tag(tag: str) -> int:
    """Drop all waiting queue items matching tag. Returns count dropped."""
    return _queue.drop_by_tag(tag)


def has_waiting_with_tag(tag: str) -> bool:
    """True if any waiting item has this tag."""
    return _queue.has_waiting_with_tag(tag)


def is_speaking() -> bool:
    """True while the worker is actively playing audio."""
    return _queue.is_speaking()


def current_audio_path() -> Optional[str]:
    """Return the direct audio file currently playing, if any."""
    return _queue.current_audio_path()
