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
from typing import Optional

logger = logging.getLogger(__name__)


# ── Queue item ─────────────────────────────────────────────────────────────────

class _Item:
    __slots__ = ("neg_priority", "seq", "text", "emotion", "audio_path", "done")

    def __init__(
        self,
        priority: int,
        seq: int,
        text: Optional[str],
        emotion: str,
        audio_path: Optional[str],
        done: threading.Event,
    ) -> None:
        self.neg_priority = -priority
        self.seq = seq
        self.text = text
        self.emotion = emotion
        self.audio_path = audio_path
        self.done = done

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

        threading.Thread(
            target=self._worker, daemon=True, name="speech-queue-worker"
        ).start()

    # ── Public API ─────────────────────────────────────────────────────────────

    def enqueue(self, text: str, emotion: str = "neutral", priority: int = 0) -> threading.Event:
        """Enqueue text for TTS. Returns an Event set when playback finishes."""
        return self._add(text, emotion, None, priority)

    def enqueue_audio_file(self, path: str, priority: int = 0) -> threading.Event:
        """Enqueue an audio file for direct playback. Returns an Event set when done."""
        return self._add(None, "neutral", path, priority)

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

    # ── Internal ───────────────────────────────────────────────────────────────

    def _add(
        self,
        text: Optional[str],
        emotion: str,
        audio_path: Optional[str],
        priority: int,
    ) -> threading.Event:
        done = threading.Event()
        should_preempt = False

        with self._not_empty:
            # Drop all waiting items of strictly lower priority
            keep = []
            for item in self._heap:
                if item.priority < priority:
                    item.done.set()
                else:
                    keep.append(item)
            if len(keep) != len(self._heap):
                self._heap = keep
                heapq.heapify(self._heap)

            # Preempt current playback if it has lower priority
            if self._speaking and self._current_priority < priority:
                should_preempt = True

            seq = self._seq
            self._seq += 1
            heapq.heappush(
                self._heap,
                _Item(priority, seq, text, emotion, audio_path, done),
            )
            self._not_empty.notify()

        if should_preempt:
            try:
                import sounddevice as sd
                sd.stop()
            except Exception:
                pass

        return done

    def _worker(self) -> None:
        while True:
            with self._not_empty:
                while not self._heap:
                    self._not_empty.wait()
                item = heapq.heappop(self._heap)

            with self._lock:
                self._speaking = True
                self._current_priority = item.priority

            try:
                try:
                    from awareness.situation import assessor as _sit
                    _sit.set_rex_speaking(True)
                except Exception:
                    pass

                if item.audio_path:
                    self._play_file(item.audio_path)
                elif item.text:
                    from audio import tts
                    tts.speak(item.text, item.emotion)
            except Exception as exc:
                logger.error("speech_queue worker error: %s", exc)
            finally:
                try:
                    from awareness.situation import assessor as _sit
                    _sit.set_rex_speaking(False)
                except Exception:
                    pass
                with self._lock:
                    self._speaking = False
                    self._current_priority = -1
                item.done.set()

    def _play_file(self, path: str) -> None:
        try:
            import numpy as np
            import sounddevice as sd
            import soundfile as sf

            from audio import echo_cancel, output_gate

            audio, samplerate = sf.read(str(path), dtype="float32", always_2d=False)
            if audio.ndim > 1:
                audio = audio.mean(axis=1)

            with output_gate.hold("speech-queue-file") as acquired:
                if not acquired:
                    logger.debug("speech_queue: playback skipped — output gate busy")
                    return
                try:
                    echo_cancel.set_playing(True)
                    sd.play(audio, samplerate)
                    sd.wait()
                finally:
                    echo_cancel.set_playing(False)
        except Exception as exc:
            logger.error("speech_queue: failed to play file %s: %s", path, exc)


# ── Module-level singleton + thin wrappers ─────────────────────────────────────

_queue = _SpeechQueue()


def enqueue(text: str, emotion: str = "neutral", priority: int = 0) -> threading.Event:
    """Enqueue text for TTS speech. Returns an Event set when playback finishes."""
    return _queue.enqueue(text, emotion, priority)


def enqueue_audio_file(path: str, priority: int = 0) -> threading.Event:
    """Enqueue an audio file for playback. Returns an Event set when done."""
    return _queue.enqueue_audio_file(path, priority)


def clear_below_priority(n: int) -> None:
    """Drop all waiting queue items with priority < n."""
    _queue.clear_below_priority(n)


def is_speaking() -> bool:
    """True while the worker is actively playing audio."""
    return _queue.is_speaking()
