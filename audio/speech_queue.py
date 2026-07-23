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
import re
import threading
import time
from pathlib import Path
from typing import Callable, Optional

logger = logging.getLogger(__name__)

_IMPERATIVE_RESPONSE_PROMPT_RE = re.compile(
    r"\b(?:give|tell|toss|send|share)\s+(?:me\s+)?(?:your\s+|a\s+)?"
    r"(?:last\s+name|surname)\b|"
    r"\b(?:last\s+name|surname)\s+(?:too|please)\b",
    re.IGNORECASE,
)


def _text_expects_reply(text: str) -> bool:
    stripped = str(text or "").strip()
    return bool(stripped and ("?" in stripped or _IMPERATIVE_RESPONSE_PROMPT_RE.search(stripped)))


def _audio_output_suppressed() -> bool:
    try:
        import config
        return bool(
            getattr(config, "NO_AUDIO_MODE", False)
            or getattr(config, "AUDIO_OUTPUT_SUPPRESSED", False)
        )
    except Exception:
        return False


def _state_suppresses_output() -> bool:
    try:
        import state as state_module
        from state import State
        # SHUTDOWN suppresses too, so a proactive/idle line that wins the race
        # against the shutdown state-flip can't get enqueued and play during the
        # power-down animation. The shutdown sign-off is spoken BEFORE the flip.
        return state_module.get_state() in (State.SLEEP, State.SHUTDOWN)
    except Exception:
        return False


def _complete_text_without_audio(
    text: Optional[str],
    done: threading.Event,
    on_start: Optional[Callable[[], None]],
) -> threading.Event:
    if on_start is not None:
        try:
            on_start()
        except Exception:
            pass
    if text:
        try:
            from utils import conv_log
            conv_log.log_rex(text)
        except Exception:
            pass
        logger.info("[speech_queue] audio suppressed — emitted text only: %r", text)
    done.set()
    return done


def _playback_handoff_options(text: Optional[str]) -> dict:
    """Return AEC stop options for queued text playback."""
    text_value = str(text or "")
    if not text_value.strip():
        return {}
    try:
        import config
        if _text_expects_reply(text_value):
            return {
                "post_playback_tail_secs": float(
                    getattr(config, "POST_QUESTION_PLAYBACK_SUPPRESSION_SECS", 0.12)
                ),
                "flush_on_playback_stop": bool(
                    getattr(config, "POST_QUESTION_FLUSH_AUDIO_BUFFER", False)
                ),
            }
        # Statements invite immediate replies too; use the short tail and skip the
        # destructive buffer flush so a reply that starts as Rex finishes survives.
        return {
            "post_playback_tail_secs": float(
                getattr(
                    config,
                    "POST_SPEECH_PLAYBACK_SUPPRESSION_SECS",
                    getattr(config, "POST_PLAYBACK_SUPPRESSION_SECS", 0.5),
                )
            ),
            "flush_on_playback_stop": bool(
                getattr(config, "POST_SPEECH_FLUSH_AUDIO_BUFFER", False)
            ),
        }
    except Exception:
        return {
            "post_playback_tail_secs": 0.5,
            "flush_on_playback_stop": True,
        }


# ── Queue item ─────────────────────────────────────────────────────────────────

class _Item:
    __slots__ = (
        "neg_priority", "seq", "text", "emotion", "audio_path",
        "done", "tag", "pre_beat_ms", "post_beat_ms", "voice_settings",
        "on_start", "log_text", "on_audio_end",
        "comedy_mode", "suppress_audio_tag", "previous_text", "voice_ref",
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
        on_start: Optional[Callable[[], None]] = None,
        log_text: bool = True,
        on_audio_end: Optional[Callable[[], None]] = None,
        comedy_mode: Optional[str] = None,
        suppress_audio_tag: bool = False,
        previous_text: Optional[str] = None,
        voice_ref: Optional[object] = None,
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
        self.on_start = on_start
        self.log_text = log_text
        self.on_audio_end = on_audio_end
        self.comedy_mode = comedy_mode
        self.suppress_audio_tag = suppress_audio_tag
        self.previous_text = previous_text
        self.voice_ref = voice_ref

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
        self._last_speech_end_at: float = 0.0  # monotonic time the last line FINISHED playing
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
        on_start: Optional[Callable[[], None]] = None,
        log_text: bool = True,
        on_audio_end: Optional[Callable[[], None]] = None,
        comedy_mode: Optional[str] = None,
        suppress_audio_tag: bool = False,
        previous_text: Optional[str] = None,
        voice_ref: Optional[object] = None,
    ) -> threading.Event:
        """Enqueue text for TTS. Returns an Event set when playback finishes.

        If tag is given, any waiting items with the same tag are dropped first —
        useful for coalescing stale presence/idle reactions.

        pre_beat_ms / post_beat_ms add a silent pause before / after speaking
        (worker holds the queue open during the beat so nothing else cuts in).

        voice_settings (optional dict of stability / style / similarity_boost /
        use_speaker_boost) overrides ElevenLabs voice parameters for this item.
        Cached separately from the default-voice take.

        voice_ref (a local_tts.VoiceRef) forces on-device synthesis in THAT voice
        — the impersonation feature uses it to clone an arbitrary person. Rex's
        normal lines pass None.

        log_text=False suppresses the per-item conversation-log/GUI write — used
        by streaming so a reply split across sentences is logged once as a turn.
        """
        return self._add(
            text, emotion, None, priority, tag,
            pre_beat_ms, post_beat_ms, voice_settings, on_start, log_text,
            on_audio_end, comedy_mode, suppress_audio_tag, previous_text, voice_ref,
        )

    def enqueue_audio_file(
        self,
        path: str,
        priority: int = 0,
        tag: Optional[str] = None,
        pre_beat_ms: int = 0,
        post_beat_ms: int = 0,
        on_start: Optional[Callable[[], None]] = None,
    ) -> threading.Event:
        """Enqueue an audio file for direct playback. Returns an Event set when done."""
        return self._add(
            None, "neutral", path, priority, tag, pre_beat_ms, post_beat_ms, None, on_start
        )

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

    def cancel_all(self) -> None:
        """Drop every queued item and interrupt current playback for shutdown."""
        with self._not_empty:
            pending = list(self._heap)
            self._heap.clear()
            for item in pending:
                item.done.set()
            speaking = self._speaking
        if speaking:
            try:
                import sounddevice as sd
                from audio import echo_cancel
                echo_cancel.request_cancel()
                sd.stop()
            except Exception:
                pass

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
        on_start: Optional[Callable[[], None]] = None,
        log_text: bool = True,
        on_audio_end: Optional[Callable[[], None]] = None,
        comedy_mode: Optional[str] = None,
        suppress_audio_tag: bool = False,
        previous_text: Optional[str] = None,
        voice_ref: Optional[object] = None,
    ) -> threading.Event:
        done = threading.Event()
        if _state_suppresses_output():
            logger.info("speech_queue: output suppressed while Rex is asleep")
            done.set()
            return done
        if _audio_output_suppressed():
            return _complete_text_without_audio(text, done, on_start)

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
                      pre_beat_ms, post_beat_ms, voice_settings, on_start, log_text,
                      on_audio_end, comedy_mode, suppress_audio_tag, previous_text,
                      voice_ref),
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

    def mark_startup_chime_played(self) -> None:
        """Prevent the automatic first-speech chime after it played elsewhere."""
        with self._lock:
            self._startup_chime_queued = True

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

            start_callbacks_fired = False

            def _fire_item_start() -> None:
                nonlocal start_callbacks_fired
                if start_callbacks_fired:
                    return
                start_callbacks_fired = True
                for _cb in _on_item_start_callbacks:
                    try:
                        _cb(item)
                    except TypeError:
                        try:
                            _cb()
                        except Exception:
                            pass
                    except Exception:
                        pass
                if item.on_start is not None:
                    try:
                        item.on_start()
                    except Exception:
                        pass

            try:
                # An item may have been popped just before shutdown cleared the heap.
                # Re-check at playback time so that race cannot start a continuation
                # after the power-down sequence has begun.
                if _state_suppresses_output():
                    logger.info("speech_queue: popped item suppressed by shutdown state")
                    continue
                try:
                    from awareness.situation import assessor as _sit
                    _sit.set_rex_speaking(True)
                except Exception:
                    pass

                # Emotion sound-effect accent: fires NOW, while the TTS below is still
                # generating (~1-2 s), so the chirp colors the reaction without ever
                # delaying it — the effect is preemptible and hands the speaker to TTS
                # the moment the synthesized audio is ready (output_gate yield hooks).
                if item.text and not item.audio_path:
                    try:
                        from audio import sound_effects
                        sound_effects.play_for_speech(item.emotion, tag=item.tag)
                    except Exception:
                        pass

                if item.pre_beat_ms > 0:
                    import time as _t
                    _t.sleep(item.pre_beat_ms / 1000.0)

                if item.audio_path:
                    self._play_file(item.audio_path, on_start=_fire_item_start)
                elif item.text:
                    from audio import tts
                    tts.speak(
                        item.text,
                        item.emotion,
                        voice_settings=item.voice_settings,
                        on_playback_start=_fire_item_start,
                        log_text=item.log_text,
                        comedy_mode=item.comedy_mode,
                        suppress_audio_tag=item.suppress_audio_tag,
                        previous_text=item.previous_text,
                        voice_ref=item.voice_ref,
                        **_playback_handoff_options(item.text),
                    )

                # The instant the spoken audio ends — BEFORE the post-line silence
                # below — a "landing" body beat hooks here so the physical button
                # lands INTO the post_beat_ms pause: line lands -> silence -> beat.
                # Must be non-blocking (animations.play_body_beat spawns its own
                # thread); a raising callback never breaks the worker.
                if item.on_audio_end is not None:
                    try:
                        item.on_audio_end()
                    except Exception:
                        pass

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
                    self._last_speech_end_at = time.monotonic()
                    self._current_priority = -1
                    self._current_audio_path = None
                item.done.set()

    def _play_file(
        self,
        path: str,
        *,
        on_start: Optional[Callable[[], None]] = None,
    ) -> None:
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
                    if on_start is not None:
                        try:
                            on_start()
                        except Exception:
                            pass
                    sd.play(audio, samplerate, blocksize=2048)
                    sd.wait()
                finally:
                    echo_cancel.set_playing(False)
        except Exception as exc:
            logger.error("speech_queue: failed to play file %s: %s", path, exc)


# ── Playback lifecycle hooks ───────────────────────────────────────────────────

_on_item_start_callbacks: list = []
_on_item_done_callbacks: list = []


def register_on_item_start(fn) -> None:
    """Register a callback invoked when a queue item starts playback."""
    if fn not in _on_item_start_callbacks:
        _on_item_start_callbacks.append(fn)


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
    on_start: Optional[Callable[[], None]] = None,
    log_text: bool = True,
    on_audio_end: Optional[Callable[[], None]] = None,
    comedy_mode: Optional[str] = None,
    suppress_audio_tag: bool = False,
    previous_text: Optional[str] = None,
    voice_ref: Optional[object] = None,
) -> threading.Event:
    """Enqueue text for TTS speech. Returns an Event set when playback finishes."""
    return _queue.enqueue(
        text, emotion, priority, tag, pre_beat_ms, post_beat_ms,
        voice_settings, on_start, log_text, on_audio_end, comedy_mode, suppress_audio_tag,
        previous_text, voice_ref,
    )


def enqueue_audio_file(
    path: str,
    priority: int = 0,
    tag: Optional[str] = None,
    pre_beat_ms: int = 0,
    post_beat_ms: int = 0,
    on_start: Optional[Callable[[], None]] = None,
) -> threading.Event:
    """Enqueue an audio file for playback. Returns an Event set when done."""
    return _queue.enqueue_audio_file(path, priority, tag, pre_beat_ms, post_beat_ms, on_start)


def reset_startup_chime_for_tests() -> None:
    _queue.reset_startup_chime_for_tests()


def mark_startup_chime_played() -> None:
    _queue.mark_startup_chime_played()


def clear_below_priority(n: int) -> None:
    """Drop all waiting queue items with priority < n."""
    _queue.clear_below_priority(n)


def cancel_all() -> None:
    """Drop all waiting speech and stop active playback immediately."""
    _queue.cancel_all()


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


def seconds_since_last_speech() -> float:
    """Monotonic seconds since Rex's last spoken line FINISHED playing (across EVERY
    path — replies, roasts, proactive — since the worker plays them all). Large when
    Rex is currently speaking or has never spoken. Lets a room reaction tell a laugh
    AT Rex's line from ambient laughter while he's been idle."""
    if _queue.is_speaking():
        return 0.0
    last = _queue._last_speech_end_at
    if last <= 0.0:
        return float("inf")
    return max(0.0, time.monotonic() - last)
