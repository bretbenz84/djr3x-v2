#!/usr/bin/env python3
"""DJ-R3X v2 — main entry point."""

import os
import sys
import argparse
from pathlib import Path
from typing import Optional

_PROJECT_ROOT = Path(__file__).resolve().parent
_PROJECT_VENV = (_PROJECT_ROOT / "venv").resolve()
_NO_AUDIO_ARGS = frozenset({"-noaudio", "--noaudio", "--no-audio"})
_NO_SERVO_ARGS = frozenset({"-noservos", "--noservos", "--no-servos"})
_LOCAL_TTS_ARGS = frozenset({"-local-tts", "--local-tts", "--localtts"})


def _seed_startup_runtime_flags(argv: list[str] | None = None) -> None:
    """Expose startup flags needed before config.py/config_loader imports."""
    args = sys.argv[1:] if argv is None else argv
    if any(arg in _NO_AUDIO_ARGS for arg in args):
        os.environ["DJR3X_NO_AUDIO_MODE"] = "1"
    # SERVOS_ENABLED is computed at config_loader import time (and imported by value
    # in hardware.servos), so --noservos must be visible as an env flag BEFORE those
    # imports — same mechanism as --noaudio above.
    if any(arg in _NO_SERVO_ARGS for arg in args):
        os.environ["DJR3X_NO_SERVOS"] = "1"
    # config.LOCAL_TTS_MODE is computed at config import time from DJR3X_LOCAL_TTS,
    # so --local-tts must be seeded here BEFORE config imports — same mechanism.
    if any(arg in _LOCAL_TTS_ARGS for arg in args):
        os.environ["DJR3X_LOCAL_TTS"] = "1"


def _verify_project_virtualenv() -> None:
    """Require the project venv interpreter before startup side effects begin."""
    running_prefix = Path(sys.prefix).resolve()
    if running_prefix == _PROJECT_VENV:
        return

    active_env = os.environ.get("VIRTUAL_ENV", "").strip() or "(none)"
    print("[FATAL] DJ-R3X must be run from this project's virtual environment.", file=sys.stderr)
    print(f"Expected venv: {_PROJECT_VENV}", file=sys.stderr)
    print(f"Current Python: {Path(sys.executable).resolve()}", file=sys.stderr)
    print(f"Current VIRTUAL_ENV: {active_env}", file=sys.stderr)
    print("Run:", file=sys.stderr)
    print("  source venv/bin/activate", file=sys.stderr)
    print("  python main.py", file=sys.stderr)
    sys.exit(1)


_verify_project_virtualenv()
_seed_startup_runtime_flags()

# Step 1: Logging must be configured before any other module logs.
from utils.logging import setup_logging, get_logger
setup_logging()
logger = get_logger(__name__)

# Step 2: Verify database schema before any memory access.
logger.info("Verifying database schema...")
try:
    from memory.database import verify_schema
    verify_schema()
except RuntimeError as e:
    print(f"[FATAL] Database not initialized: {e}", file=sys.stderr)
    print("Run:  python setup_assets.py", file=sys.stderr)
    sys.exit(1)

# Rex's episodic-memory DB (rex.db) — create/ensure the schema (best-effort; never
# fatal). Phase 1: capture only, nothing reads it back yet.
try:
    from memory import episodes as _episodes
    _episodes.ensure_ready()
except Exception as _exc:
    logger.debug("rex.db ensure_ready failed: %s", _exc)

# Step 3: Load config — raises RuntimeError at import time if API keys are missing.
logger.info("Loading configuration and API keys...")
try:
    from utils import config_loader  # noqa: F401  triggers import-time key validation
except RuntimeError as e:
    logger.critical("Configuration error — missing or invalid API key: %s", e)
    sys.exit(1)

# All remaining imports after config is confirmed valid.
import time
import signal
import threading
import random

import numpy as np
import sounddevice as sd
import soundfile as sf

import config
import state
from state import State
from hardware import servos, leds_head, leds_chest, motion
from utils.config_loader import (
    SERVOS_ENABLED,
    HEAD_LEDS_ENABLED,
    CHEST_LEDS_ENABLED,
    MOTION_PORT_SET,
    CAMERA_ENABLED,
    CAMERA_SELECTION_DESCRIPTION,
    AUDIO_ENABLED,
    AUDIO_SELECTION_DESCRIPTION,
)
from sequences import animations
from utils import phrase_cycler
from audio import (
    stream,
    scene as audio_scene,
    output_gate,
    echo_cancel,
    tts,
    speech_queue,
    speaker_id,
    transcription,
)
from vision import camera, scene as vision_scene, face_expression, animal_detector, pose
from perception import place_service
from awareness import chronoception, interoception
from intelligence import consciousness, emotion_orchestrator, interaction, local_llm, motion_controller


def _verify_local_whisper_model() -> None:
    """Fail fast if setup_assets.py has not downloaded the local Whisper model."""
    model_config = (
        Path(__file__).resolve().parent / config.WHISPER_MODEL_DIR / "config.json"
    )
    if not model_config.exists():
        print(
            f"[FATAL] Local Whisper model not found: {model_config}",
            file=sys.stderr,
        )
        print("Run:  python setup_assets.py", file=sys.stderr)
        sys.exit(1)


def _audio_file_match_keys(path: str) -> set[str]:
    path_obj = Path(path)
    return {
        path_obj.name.lower(),
        str(path_obj).lower(),
        str(path_obj.as_posix()).lower(),
    }


def _is_speech_animated_audio_file(path: str) -> bool:
    candidates = _audio_file_match_keys(path)
    configured = {
        str(item).strip().lower()
        for item in getattr(config, "SPEECH_ANIMATED_AUDIO_FILES", [])
        if str(item).strip()
    }
    return any(item in candidates for item in configured)


def _is_listening_chime_audio_file(path: str) -> bool:
    candidates = _audio_file_match_keys(path)
    configured = str(getattr(config, "LISTENING_CHIME_FILE", "") or "").strip().lower()
    return bool(configured and configured in candidates)


def _conversation_line_for_audio_file(path: str) -> Optional[str]:
    candidates = _audio_file_match_keys(path)
    configured = getattr(config, "SPEECH_ANIMATED_AUDIO_TRANSCRIPTS", {}) or {}
    if not isinstance(configured, dict):
        return None
    for key, text in configured.items():
        if str(key).strip().lower() not in candidates:
            continue
        line = str(text or "").strip()
        return line or None
    return None


def _log_conversation_line_for_audio_file(path: str) -> None:
    line = _conversation_line_for_audio_file(path)
    if not line:
        return
    try:
        from utils import conv_log
        conv_log.log_rex(line)
    except Exception as exc:
        logger.debug("direct clip conversation log failed for %s: %s", path, exc)


def _drive_speech_clip_outputs(
    audio: np.ndarray,
    samplerate: int,
    stop_event: threading.Event,
) -> None:
    interval = float(getattr(config, "TTS_LED_UPDATE_INTERVAL_SECS", 0.04) or 0.04)
    chunk_len = max(1, int(int(samplerate) * interval))
    min_delta = int(getattr(config, "HEAD_LED_SPEAK_LEVEL_MIN_DELTA", 8))
    last_sent = -1
    for i in range(0, len(audio), chunk_len):
        if stop_event.is_set():
            break
        chunk = audio[i:i + chunk_len]
        if chunk.size == 0:
            continue
        rms = float(np.sqrt(np.mean(chunk * chunk)))
        brightness = min(255, int(rms * config.TTS_LED_BRIGHTNESS_SCALE))
        # Throttle the mouth-level flood (see audio/tts._drive_leds): fewer serial
        # writes during speech = fewer dropped commands on the lossy head link.
        if (
            last_sent < 0
            or abs(brightness - last_sent) >= min_delta
            or (brightness == 0 and last_sent != 0)
        ):
            leds_head.speak_level(brightness)
            last_sent = brightness
        servos.speech_reactive_move(brightness / 255.0)
        stop_event.wait(timeout=interval)


def _play_audio_file(
    path: str,
    *,
    speech_animation: Optional[bool] = None,
    emotion: str = "neutral",
) -> None:
    """Play a pre-recorded audio file synchronously via sounddevice.

    Using sounddevice (PortAudio) here — not pygame.mixer (SDL) — keeps all
    playback on a single backend. Concurrent SDL + PortAudio on macOS produces
    PaMacCore err=-50 glitches, which surface as choppy clip playback and
    duplicated/echoed TTS output.
    """
    if bool(
        getattr(config, "NO_AUDIO_MODE", False)
        or getattr(config, "AUDIO_OUTPUT_SUPPRESSED", False)
    ):
        logger.info("Audio suppressed; skipping direct playback: %s", path)
        return
    with output_gate.hold("startup_or_shutdown_clip") as acquired:
        if not acquired:
            return
        animate_speech = (
            _is_speech_animated_audio_file(path)
            if speech_animation is None
            else bool(speech_animation)
        )
        audio, samplerate = sf.read(path, dtype="float32", always_2d=False)
        if audio.ndim > 1:
            audio = audio.mean(axis=1).astype(np.float32)
        audio = np.asarray(audio, dtype=np.float32)

        gain = float(getattr(config, "STARTUP_SHUTDOWN_AUDIO_GAIN", 0.65) or 0.0)
        if gain != 1.0:
            audio = audio * max(0.0, gain)

        fade_samples = min(int(0.015 * int(samplerate)), max(0, audio.size // 2))
        if fade_samples > 1:
            fade = np.linspace(0.0, 1.0, fade_samples, dtype=np.float32)
            audio[:fade_samples] *= fade
            audio[-fade_samples:] *= fade[::-1]

        peak_limit = float(getattr(config, "STARTUP_SHUTDOWN_AUDIO_PEAK_LIMIT", 0.80) or 0.80)
        peak = float(np.max(np.abs(audio))) if audio.size else 0.0
        if peak > peak_limit > 0.0:
            audio = audio * (peak_limit / peak)

        stop_event = threading.Event()
        led_thread = None
        expected_duration = len(audio) / float(samplerate) if samplerate else 0.0
        play_started_at = time.monotonic()
        emotion_frame = emotion_orchestrator.frame_for_speech(emotion)
        led_emotion = emotion_frame.led_style
        try:
            if animate_speech:
                emotion_orchestrator.publish_frame(
                    emotion_frame,
                    ttl_secs=max(2.0, expected_duration + 2.0),
                )
                try:
                    animations.speech_activity_start()
                    servos.begin_speech_motion(emotion_frame)
                except Exception as exc:
                    logger.debug("direct clip speech servo start failed: %s", exc)
                try:
                    from awareness.situation import assessor as _sit
                    _sit.set_rex_speaking(True)
                except Exception:
                    pass
                leds_head.speak(led_emotion)
                # Re-assert the eyes ON every clip — the mouth SPEAK command never
                # touches the eyes, and the heartbeat keeps them lit between turns.
                leds_head.ensure_eyes_on(led_emotion)
                leds_chest.speak(led_emotion)
                led_thread = threading.Thread(
                    target=_drive_speech_clip_outputs,
                    args=(audio, int(samplerate), stop_event),
                    daemon=True,
                    name="direct-clip-leds",
                )
            echo_cancel.set_playing(True)
            echo_cancel.add_reference(audio)
            if led_thread is not None:
                led_thread.start()
            sd.play(audio, samplerate, **tts.playback_stream_kwargs())
            _log_conversation_line_for_audio_file(path)
            sd.wait()
        finally:
            elapsed = time.monotonic() - play_started_at
            remaining = expected_duration - elapsed
            if remaining > 0.05 and not echo_cancel.was_canceled():
                logger.warning(
                    "direct clip sd.wait() returned %.2fs early — "
                    "holding speech/AEC outputs for remaining %.2fs",
                    remaining,
                    remaining,
                )
                time.sleep(remaining)
            stop_event.set()
            if led_thread is not None and led_thread.is_alive():
                led_thread.join(timeout=1.0)
            if animate_speech:
                shutdown_now = _is_shutdown_state()
                try:
                    if shutdown_now:
                        leds_head.off()
                    else:
                        leds_head.speak_stop()
                except Exception as exc:
                    logger.warning("direct clip head LED cleanup failed: %s", exc)
                try:
                    if shutdown_now:
                        leds_chest.off()
                    else:
                        leds_chest.active()
                except Exception as exc:
                    logger.debug("direct clip chest LED cleanup failed: %s", exc)
                try:
                    servos.end_speech_motion()
                except Exception as exc:
                    logger.debug("direct clip speech servo stop failed: %s", exc)
                try:
                    animations.speech_activity_stop()
                except Exception as exc:
                    logger.debug("direct clip speech activity clear failed: %s", exc)
                try:
                    from awareness.situation import assessor as _sit
                    _sit.set_rex_speaking(False)
                except Exception:
                    pass
            echo_cancel.set_playing(False)


def _play_listening_chime_async(reason: str) -> None:
    """Queue the listening chime through speech_queue so AEC suppresses it."""
    if bool(
        getattr(config, "NO_AUDIO_MODE", False)
        or getattr(config, "AUDIO_OUTPUT_SUPPRESSED", False)
    ):
        return
    if not bool(getattr(config, "PLAY_LISTENING_CHIME", True)):
        return
    path = Path(getattr(config, "LISTENING_CHIME_FILE", "") or "")
    if not path.is_absolute():
        path = Path(__file__).resolve().parent / path
    if not path.exists():
        logger.warning("Listening chime missing: %s", path)
        return
    try:
        logger.info("Playing listening chime (%s): %s", reason, path)
        speech_queue.enqueue_audio_file(
            str(path),
            priority=1,
            tag="system:listening_chime",
        )
    except Exception as exc:
        logger.warning("Could not queue listening chime (%s): %s", reason, exc)


def _configured_sensor_warning_lines(kind: str, fallback: list[str]) -> list[str]:
    configured = getattr(config, "STARTUP_SENSOR_WARNING_LINES", {}) or {}
    raw_lines = None
    if isinstance(configured, dict):
        raw_lines = configured.get(kind)
    if isinstance(raw_lines, str):
        raw_lines = [raw_lines]
    if not raw_lines:
        raw_lines = fallback
    lines = [str(line).strip() for line in raw_lines if str(line).strip()]
    return lines or fallback


def _startup_device_warning_line(
    *,
    camera_available: bool,
    audio_available: bool,
) -> Optional[str]:
    """Pick a startup warning when camera and/or microphone input is unavailable."""
    if camera_available and audio_available:
        return None

    if not camera_available and not audio_available:
        lines = _configured_sensor_warning_lines(
            "both",
            [
                "Optical sensors and audio receptors are both offline. Great. No scans, no comms, and somehow I am still expected to look professional.",
            ],
        )
    elif not camera_available:
        lines = _configured_sensor_warning_lines(
            "camera",
            [
                "Optical sensors are offline. Wonderful. I will navigate by vibes and whatever the navicomputer calls plausible.",
            ],
        )
    else:
        lines = _configured_sensor_warning_lines(
            "audio",
            [
                "Audio receptors are offline. Terrific. Please submit all brilliant organic commentary by datapad, preferably spell-checked.",
            ],
        )
    return random.choice(lines)


def _is_shutdown_state() -> bool:
    try:
        return state.is_state(State.SHUTDOWN)
    except Exception:
        return False


class _StartupAborted(Exception):
    """Shutdown was requested while controller startup was still running.

    Raised at the cooperative checkpoints inside _run_controller_startup so a
    GUI-mode boot (which runs on a background thread) stops at the next phase
    boundary when the user closes the window or hits Ctrl+C, instead of
    finishing the whole boot — audio lines, animations, service starts — with
    no window on screen."""


def _abort_startup_if_shutdown(stage: str) -> None:
    if _is_shutdown_state():
        logger.info("Shutdown requested during startup — aborting before %s.", stage)
        raise _StartupAborted(stage)


def _turn_leds_off_for_shutdown() -> None:
    """Final LED darkening on shutdown — a smooth fade-to-black (not an instant off)
    for a lifelike power-down. The firmware runs the ~4s ramp autonomously, so this
    returns immediately and the fade finishes well before the serial ports close.
    Idempotent at the firmware level, so the repeat calls on the shutdown path are
    harmless."""
    try:
        leds_head.fade_off()
    except Exception as exc:
        logger.warning("Could not fade head LEDs off during shutdown: %s", exc)
    try:
        leds_chest.fade_off()
    except Exception as exc:
        logger.warning("Could not fade chest LEDs off during shutdown: %s", exc)


def _queue_startup_device_warning(
    *,
    camera_available: bool,
    audio_available: bool,
) -> Optional[str]:
    """Queue the in-character startup self-diagnostic warning, if needed."""
    if not bool(getattr(config, "STARTUP_SENSOR_WARNING_ENABLED", True)):
        return None
    if bool(
        getattr(config, "NO_AUDIO_MODE", False)
        or getattr(config, "AUDIO_OUTPUT_SUPPRESSED", False)
    ):
        return None

    line = _startup_device_warning_line(
        camera_available=camera_available,
        audio_available=audio_available,
    )
    if not line:
        return None

    logger.warning("Startup sensor warning queued: %s", line)
    emotion = str(getattr(config, "STARTUP_SENSOR_WARNING_EMOTION", "curious") or "curious")
    speech_queue.enqueue(
        line,
        emotion,
        priority=1,
        tag="startup:sensor_warning",
    )
    return line


def _queue_camera_reconnect_line(downtime_secs: float = 0.0) -> Optional[str]:
    """Queue a short in-character line when camera frames recover after outage."""
    if not bool(getattr(config, "CAMERA_RECONNECT_TTS_ENABLED", True)):
        return None
    if bool(
        getattr(config, "NO_AUDIO_MODE", False)
        or getattr(config, "AUDIO_OUTPUT_SUPPRESSED", False)
    ):
        return None
    try:
        if state.get_state() in (State.QUIET, State.SLEEP, State.SHUTDOWN):
            return None
    except Exception:
        return None

    min_downtime = max(
        0.0,
        float(getattr(config, "CAMERA_RECONNECT_TTS_MIN_DOWNTIME_SECS", 1.0) or 0.0),
    )
    if float(downtime_secs or 0.0) < min_downtime:
        return None

    lines = [
        str(line).strip()
        for line in getattr(config, "CAMERA_RECONNECT_TTS_LINES", [])
        if str(line).strip()
    ]
    if not lines:
        return None

    global _last_camera_reconnect_line
    candidates = [line for line in lines if line != _last_camera_reconnect_line] or lines
    line = random.choice(candidates)
    _last_camera_reconnect_line = line
    emotion = str(getattr(config, "CAMERA_RECONNECT_TTS_EMOTION", "happy") or "happy")
    logger.info("Camera reconnect TTS queued after %.1fs offline: %s", downtime_secs, line)
    speech_queue.enqueue(
        line,
        emotion,
        priority=0,
        tag="camera:reconnected",
    )
    return line


def _place_event_sink(name: str, payload: dict) -> None:
    """Voice feedback for place-recognition edge events (perception/place_service).

    Two events speak; the rest stay in the log:
      - enrollment_failed: Rex acknowledged a room-name tell with "Got it — I'll
        remember this place", so a capture that later dies (someone stood in front of
        the lens for the whole window) must be owned out loud.
      - possible_duplicate_place at TWIN similarity (>= PLACE_TWIN_WARN_TTS_SIM): the
        service auto-commits the room (the human's explicit name wins), but a gallery
        born near-identical to another room's WILL be confused with it — say so and
        invite a walkaround instead of failing silently later.
    Success is silent (the ack already covered it)."""
    try:
        if name == "enrollment_failed":
            if not bool(getattr(config, "PLACE_ENROLL_FAIL_TTS_ENABLED", True)):
                return
            lines = getattr(config, "PLACE_ENROLL_FAIL_TTS_LINES", [])
            tag = "place:enroll_failed"
            fmt = {}
            log_line = (
                f"Place enrollment failed (room={(payload or {}).get('name')!r}, "
                f"collected={(payload or {}).get('collected')})"
            )
        elif name == "possible_duplicate_place":
            sim = float((payload or {}).get("similarity") or 0.0)
            threshold = float(getattr(config, "PLACE_TWIN_WARN_TTS_SIM", 0.95))
            if sim < threshold:
                return
            lines = getattr(config, "PLACE_TWIN_WARN_TTS_LINES", [])
            tag = "place:twin_warning"
            fmt = {
                "new": str((payload or {}).get("new_place") or "this room"),
                "existing": str((payload or {}).get("existing_place") or "another room"),
            }
            log_line = (
                f"Place twins (sim={sim:.2f}): {fmt['new']!r} vs {fmt['existing']!r}"
            )
        else:
            return

        if bool(
            getattr(config, "NO_AUDIO_MODE", False)
            or getattr(config, "AUDIO_OUTPUT_SUPPRESSED", False)
        ):
            return
        if state.get_state() in (State.QUIET, State.SLEEP, State.SHUTDOWN):
            return
        lines = [str(line).strip() for line in lines if str(line).strip()]
        if not lines:
            return
        line = random.choice(lines)
        if fmt:
            line = line.format(**fmt)
        logger.info("%s — queueing line: %s", log_line, line)
        speech_queue.enqueue(line, "neutral", priority=0, tag=tag)
    except Exception as exc:
        logger.debug("place event sink failed: %s", exc)


def _episodic_shutdown_summary() -> None:
    """Summarize this session's conversation via the LLM and store it as a
    'conversation_summary' episode in rex.db. Called at shutdown BEFORE interaction.stop()
    so the transcript + session people are still intact. Failure-safe and TIMEOUT-bounded
    (a background thread + join) so a slow/hung LLM call can never block shutdown."""
    try:
        import config
        if not getattr(config, "EPISODIC_SHUTDOWN_SUMMARY_ENABLED", True):
            return
        from memory import episodes
        if episodes._suppressed():  # gated: disabled / under the test runner
            return
        from memory import conversations as conv_memory
        transcript = conv_memory.get_session_transcript() or []
        if not transcript:
            return  # nothing happened this session
        # Substance pre-gate: a diary-worthy session needs a real exchange. Short
        # test/command sessions used to burn an LLM call AND leave a "nothing
        # happened" row (field 2026-07-17: a third of the diary was null reports).
        rex_labels = {"rex", "dj-r3x", "djr3x", "dj r3x", "r3x", "dj-rex", "assistant"}
        human_turns = sum(
            1 for t in transcript
            if str(t.get("speaker") or t.get("role") or "").strip().lower() not in rex_labels
        )
        min_turns = int(getattr(config, "EPISODIC_SUMMARY_MIN_HUMAN_TURNS", 3))
        if human_turns < min_turns:
            logger.info("Diary: session below substance gate (%d/%d human turns) — no entry.",
                        human_turns, min_turns)
            return

        # Best-effort: who was around (soft person refs for the episode).
        people = []
        try:
            from intelligence import interaction as _intx
            from memory import people as _people
            for pid in list(getattr(_intx, "_session_person_ids", set()) or set()):
                if isinstance(pid, int):
                    try:
                        name = (_people.get_person(pid) or {}).get("name")
                    except Exception:
                        name = None
                    people.append({"person_id": pid, "name": name})
        except Exception:
            pass

        result: dict = {}

        def _work() -> None:
            try:
                from intelligence import llm
                names = [p.get("name") for p in people if p.get("name")]
                result["entry"] = llm.generate_diary_entry(transcript, people_names=names)
            except Exception as exc:
                logger.debug("episodic shutdown summary llm failed: %s", exc)

        timeout = float(getattr(config, "EPISODIC_SHUTDOWN_SUMMARY_TIMEOUT_SECS", 12.0))
        worker = threading.Thread(target=_work, daemon=True, name="episodic-shutdown-summary")
        worker.start()
        worker.join(timeout)
        entry = result.get("entry") or {}
        note = (entry.get("note") or "").strip()
        min_sal = float(getattr(config, "EPISODIC_SUMMARY_MIN_SALIENCE", 0.3))
        if not entry.get("remember") or not note:
            logger.info("Diary: extractor judged this session not memorable — no entry.")
        elif entry.get("salience", 0.0) < min_sal:
            logger.info("Diary: salience %.2f below floor %.2f — no entry.",
                        entry.get("salience", 0.0), min_sal)
        else:
            detail = {"people": people} if people else {}
            if entry.get("open_threads"):
                detail["open_threads"] = entry["open_threads"]
            episodes.record_conversation_summary(
                note, people=people or None,
                salience=float(entry["salience"]), detail=detail or None,
            )
            logger.info("Diary: entry saved (salience %.2f, %d open thread(s), %d people).",
                        entry["salience"], len(entry.get("open_threads") or []), len(people))
    except Exception as exc:
        logger.debug("episodic shutdown summary failed: %s", exc)


def _shutdown() -> None:
    logger.info("=== Shutdown sequence begin ===")

    # GUI/button shutdown can arrive while a multi-sentence reply is streaming.
    # Flush both the active sentence and queued continuations before the power-down
    # clip claims the speaker; otherwise TTS can continue over the shutdown sound
    # and servo droop.
    try:
        speech_queue.cancel_all()
    except Exception as exc:
        logger.debug("speech cancellation at shutdown failed: %s", exc)

    # ── Power-down theatrics FIRST, so they fire the INSTANT shutdown begins ─────
    # The LED fade, shutdown sound, and servo droop all kick off right after Rex's
    # sign-off. The session save + the rest of the service teardown then run WHILE
    # these play out (none of them touch the LEDs / audio / servos), instead of
    # making the user wait ~2-3s for the bookkeeping before anything happens.
    # Hardware is closed only once the theatrics finish (below).
    #
    # Stop the head-driving loop first so face-tracking can't fight the droop — it's
    # a fast stop (the loop waits on _stop_event, so it exits immediately). The arm
    # wander already idles itself in SHUTDOWN; breathing stops inside
    # animations.shutdown().
    logger.info("Stopping intelligence.consciousness...")
    try:
        consciousness.stop()
    except Exception as exc:
        logger.warning("consciousness stop failed: %s", exc)

    logger.info("Starting synchronized power-down (LED fade + audio + servo droop)...")
    powerdown_started = time.monotonic()
    _turn_leds_off_for_shutdown()  # start the ~4s LED fade NOW, in lockstep with the rest

    _audio_thread = None
    if config.PLAY_SHUTDOWN_AUDIO:
        logger.info("Playing shutdown audio: %s", config.SHUTDOWN_AUDIO_FILE)
        def _play_shutdown_audio() -> None:
            try:
                _play_audio_file(config.SHUTDOWN_AUDIO_FILE)
            except Exception as e:
                logger.warning("Could not play shutdown audio: %s", e)
        _audio_thread = threading.Thread(target=_play_shutdown_audio, daemon=True, name="shutdown_audio")
        _audio_thread.start()
    else:
        logger.info("Shutdown audio disabled by config.PLAY_SHUTDOWN_AUDIO")

    logger.info("Playing shutdown animation...")
    animations.shutdown()  # servo droop (~2s); the LED fade keeps ramping in firmware

    # ── Session save + remaining teardown — runs WHILE the sound / LED fade finish ──
    # Save Rex's current preoccupation so it resumes next launch (best-effort).
    try:
        from intelligence import rex_pov
        rex_pov.persist()
    except Exception as exc:
        logger.debug("rex_pov persist on shutdown failed: %s", exc)

    # Episodic memory: summarize this session into rex.db BEFORE interaction.stop()
    # clears the transcript. Timeout-bounded so it can't hang shutdown.
    _episodic_shutdown_summary()

    # Consolidation sweep: per-kind diary retention (person_seen dedup/age-out,
    # visit ageing, stale pending room questions) — pure SQL, includes the scene
    # prune below via episodic_recall.prune().
    try:
        from memory import consolidation
        consolidation.run()
    except Exception as exc:
        logger.debug("consolidation sweep failed: %s", exc)

    # Retention: cap the diary's scene episodes so they don't accumulate unbounded
    # (~15/run, only ever used as a clustered "vibe"). Best-effort, gated internally.
    try:
        from memory import episodic_recall
        pruned = episodic_recall.prune()
        if pruned:
            logger.info("Pruned %d old scene episode(s) from rex.db.", pruned)
    except Exception as exc:
        logger.debug("episodic prune on shutdown failed: %s", exc)

    # Stop the remaining services in reverse startup order.
    logger.info("Stopping intelligence.interaction...")
    interaction.stop()  # also calls wake_word.stop() internally

    logger.info("Unloading local LLM...")
    local_llm.unload()

    logger.info("Stopping awareness.interoception...")
    interoception.stop()

    logger.info("Stopping awareness.chronoception...")
    chronoception.stop()

    logger.info("Stopping vision.scene...")
    vision_scene.stop()

    logger.info("Stopping perception.place_service...")
    place_service.stop()

    logger.info("Stopping vision.animal_detector...")
    animal_detector.stop()

    logger.info("Stopping vision.face_expression...")
    face_expression.stop()

    logger.info("Stopping vision.pose...")
    pose.stop()

    logger.info("Stopping vision.camera...")
    camera.stop()

    logger.info("Stopping audio.scene...")
    audio_scene.stop()

    logger.info("Stopping audio.stream...")
    stream.stop()

    # Let the shutdown clip finish (its ~5s also covers the ~4s LED fade) before we
    # close the serial ports, so neither the sound nor the fade is cut short. Even
    # with no audio, hold for the remainder of the LED fade so it doesn't snap off.
    if _audio_thread is not None:
        _audio_thread.join()
    fade_remaining = float(getattr(config, "SHUTDOWN_LED_FADE_SECS", 4.0)) - (
        time.monotonic() - powerdown_started
    )
    if fade_remaining > 0:
        time.sleep(fade_remaining)

    # Close hardware.
    logger.info("Closing hardware...")
    servos.shutdown()
    leds_head.disconnect()
    leds_chest.disconnect()
    motion_controller.disconnect()   # stops heartbeat + leaves the base stopped

    # Release the single-instance lock so the always-on supervisor can relaunch
    # on the next "wake up rex". (The OS would also free it on process exit.)
    try:
        from utils import single_instance
        single_instance.release()
    except Exception as exc:
        logger.debug("single-instance release skipped: %s", exc)

    logger.info("=== Shutdown complete ===")


_gui_bridge_stop = threading.Event()
_gui_bridge_thread: threading.Thread | None = None
_last_camera_reconnect_line: Optional[str] = None


def _audio_output_suppressed() -> bool:
    return bool(
        getattr(config, "NO_AUDIO_MODE", False)
        or getattr(config, "AUDIO_OUTPUT_SUPPRESSED", False)
    )


def _configure_audio_output_device() -> None:
    """Route all sounddevice playback to a configured output device.

    Used to send Rex's audio THROUGH the ReSpeaker Lite so its onboard hardware AEC
    receives the playback reference and can cancel his voice from the mic. No-op when
    neither AUDIO_OUTPUT_DEVICE_INDEX (>=0) nor AUDIO_OUTPUT_DEVICE_NAME is set —
    playback then uses the OS default output, unchanged.
    """
    idx = int(getattr(config, "AUDIO_OUTPUT_DEVICE_INDEX", -1))
    name = str(getattr(config, "AUDIO_OUTPUT_DEVICE_NAME", "") or "").strip()
    if idx < 0 and not name:
        return
    try:
        import sounddevice as sd
        if idx < 0 and name:
            for i, dev in enumerate(sd.query_devices()):
                if name.lower() in str(dev.get("name", "")).lower() and int(dev.get("max_output_channels", 0)) > 0:
                    idx = i
                    break
        if idx is None or idx < 0:
            logger.warning(
                "Audio output device %r not found; using default output.",
                name or idx,
            )
            return
        current = sd.default.device
        in_dev = current[0] if isinstance(current, (list, tuple)) else current
        sd.default.device = (in_dev, idx)
        logger.info(
            "Audio output routed to device %d (%s) for hardware AEC.",
            idx, sd.query_devices(idx).get("name", "?"),
        )
    except Exception as exc:
        logger.warning("Could not set audio output device: %s", exc)


def _select_cycling_tts_line(lines, state_path: Path) -> str:
    """Pick a line, cycling without back-to-back repeats. See utils.phrase_cycler."""
    return phrase_cycler.select_cycling_line(lines, state_path)


def _select_startup_boot_tts_line() -> str:
    """Pick a boot ('still loading') filler line, cycling without back-to-back repeats."""
    state_path = Path(
        str(getattr(config, "STARTUP_BOOT_TTS_STATE_PATH", "") or "")
        or (Path(__file__).resolve().parent / "assets" / "state" / "startup_boot_tts.json")
    )
    line = _select_cycling_tts_line(getattr(config, "STARTUP_BOOT_TTS_LINES", None), state_path)
    return line or str(getattr(config, "STARTUP_BOOT_TTS_LINE", "") or "").strip()


def _select_startup_ready_tts_line() -> str:
    """Pick a 'models loaded, I'm ready' roast line, cycling without back-to-back repeats."""
    state_path = Path(
        str(getattr(config, "STARTUP_READY_TTS_STATE_PATH", "") or "")
        or (Path(__file__).resolve().parent / "assets" / "state" / "startup_ready_tts.json")
    )
    return _select_cycling_tts_line(getattr(config, "STARTUP_READY_TTS_LINES", None), state_path)


# Handle for the looping startup "thinking" effect (started after the boot line,
# stopped just before the ready line so it never talks under it).
_startup_thinking_loop = None


def _start_startup_boot_tts_thread(
    wait_for: threading.Thread | None = None,
) -> threading.Thread | None:
    if _audio_output_suppressed():
        logger.info("Startup boot TTS disabled by --noaudio")
        return None
    if not bool(getattr(config, "PLAY_STARTUP_BOOT_TTS", True)):
        logger.info("Startup boot TTS disabled by config.PLAY_STARTUP_BOOT_TTS")
        return None

    line = _select_startup_boot_tts_line()
    if not line:
        logger.info("Startup boot TTS skipped because no boot line is configured")
        return None

    try:
        delay_secs = float(getattr(config, "STARTUP_BOOT_TTS_DELAY_SECS", 1.5) or 0.0)
    except (TypeError, ValueError):
        delay_secs = 1.5
    delay_secs = max(0.0, delay_secs)
    emotion = str(getattr(config, "STARTUP_BOOT_TTS_EMOTION", "curious") or "curious")

    def _speak_startup_boot_line() -> None:
        try:
            # Pre-fetch/cache the audio up front so the actual playback is a cheap
            # cache hit even though it now overlaps the CPU-heavy model preloads (the
            # network fetch + decode happens here, before the loads peak). Cuts the
            # "audio thread starved during Whisper load" glitch the old serial order
            # was avoiding.
            try:
                tts.ensure_cached(line, emotion=emotion)
            except Exception as exc:
                logger.debug("startup boot TTS pre-cache skipped: %s", exc)
            # Chain straight onto the startup audio clips: the cache fetch above
            # ran WHILE the mp3s played, and this join releases the moment the
            # last clip finishes — so the boot line follows the spoken clip after
            # only the configured beat, instead of sitting silent through the
            # rest of the blocking startup servo animation + service starts
            # (~8s of dead air in live runs).
            if wait_for is not None and wait_for.is_alive():
                wait_for.join()
            if delay_secs > 0.0:
                time.sleep(delay_secs)
            if _audio_output_suppressed():
                return
            if _is_shutdown_state():
                logger.info("Skipping startup boot TTS — shutdown requested mid-boot.")
                return
            logger.info("Playing startup boot TTS after %.1fs delay: %s", delay_secs, line)
            tts.speak(line, emotion)
            if not _is_shutdown_state():
                # Fill the remaining model-warmup gap before the ready line. LOOPED,
                # not one-shot: the clip is ~1.5 s and the wait it covers is many
                # times that, so a single play left most of the gap silent (owner
                # 2026-07-24). Gated, so it stays preemptible and the ready line
                # takes the speaker the moment startup is genuinely complete; it is
                # also stopped explicitly before that line (see _startup_thinking_loop).
                try:
                    from audio import sound_effects
                    global _startup_thinking_loop
                    _startup_thinking_loop = sound_effects.start_loop(
                        "thinking",
                        gap_secs=float(getattr(config, "STARTUP_THINKING_LOOP_GAP_SECS", 1.2)),
                        max_secs=float(getattr(config, "STARTUP_THINKING_LOOP_MAX_SECS", 90.0)),
                    )
                except Exception as exc:
                    logger.debug("startup thinking effect skipped: %s", exc)
        except Exception as exc:
            logger.warning("Could not play startup boot TTS: %s", exc)

    thread = threading.Thread(
        target=_speak_startup_boot_line,
        daemon=True,
        name="startup_boot_tts",
    )
    thread.start()
    return thread


def _apply_startup_mode_overrides(*, jeopardy: bool = False) -> None:
    if not jeopardy:
        return
    logger.info("Startup mode: Jeopardy requested; suppressing startup introductions.")
    setattr(config, "STARTUP_GROUP_GREETING_ENABLED", False)
    setattr(config, "MOOD_AWARE_FIRST_SIGHT_ENABLED", False)
    try:
        state.set_state(State.ACTIVE)
    except Exception as exc:
        logger.debug("Could not pre-set ACTIVE state for Jeopardy startup: %s", exc)


def _launch_startup_jeopardy() -> None:
    logger.info("Launching Jeopardy startup mode.")
    try:
        from features import games as games_mod
        from memory import conversations as conv_memory
        from utils import conv_log

        response = games_mod.start_game("jeopardy")
        if not response:
            logger.warning("Jeopardy startup returned no opening response.")
            return
        try:
            intro_path = Path(getattr(config, "JEOPARDY_AUDIO_DIR", "assets/audio/jeopardy")) / "jeopardy-intro.mp3"
            current_audio = speech_queue.current_audio_path() or ""
            intro_is_playing = Path(current_audio).name == intro_path.name if current_audio else False
            intro_is_waiting = speech_queue.has_waiting_with_tag("jeopardy:intro")
            if intro_path.exists() and not intro_is_playing and not intro_is_waiting:
                logger.info("Jeopardy startup intro was not queued; queueing %s", intro_path)
                speech_queue.enqueue_audio_file(
                    str(intro_path),
                    priority=1,
                    tag="jeopardy:intro",
                )
        except Exception as exc:
            logger.debug("Could not verify Jeopardy startup intro audio: %s", exc)

        try:
            conv_memory.add_to_transcript("Rex", response)
        except Exception as exc:
            logger.debug("Could not add Jeopardy startup line to transcript: %s", exc)
        try:
            conv_log.log_rex(response)
        except Exception as exc:
            logger.debug("Could not add Jeopardy startup line to GUI conversation log: %s", exc)
        try:
            consciousness.note_rex_utterance(
                response,
                wait_secs=float(getattr(config, "QUESTION_RESPONSE_WAIT_SECS", 20.0)),
            )
        except Exception as exc:
            logger.debug("Could not register Jeopardy startup utterance: %s", exc)

        speech_queue.enqueue(response, "excited", priority=1, tag="startup:jeopardy")
    except Exception as exc:
        logger.exception("Failed to launch Jeopardy startup mode: %s", exc)


def _run_controller_startup(*, startup_jeopardy: bool = False) -> None:
    _apply_startup_mode_overrides(jeopardy=startup_jeopardy)
    no_audio = bool(getattr(config, "NO_AUDIO_MODE", False))

    # Serialize sounddevice play/stop before ANY playback so wake-word barge-in
    # (a cross-thread stop() + immediate replay) can't crash the global CoreAudio
    # stream. Must run before the startup clip — the first sd.play(). See sd_guard.
    try:
        from audio import sd_guard
        sd_guard.install()
    except Exception as exc:
        logger.debug("sd_guard install skipped: %s", exc)

    # Route playback to a specific output device (e.g. the ReSpeaker Lite, so its
    # onboard hardware AEC gets the reference). No-op unless configured.
    if not no_audio:
        _configure_audio_output_device()

    if no_audio:
        logger.info("Skipping local Whisper verification (--noaudio text-input mode).")
    else:
        logger.info("Verifying local Whisper model...")
        _verify_local_whisper_model()

    # Step 4: Initialize hardware and log enabled/disabled status.
    logger.info("=== Initializing hardware ===")

    servo_ok = servos.connect()
    if SERVOS_ENABLED and servo_ok:
        logger.info("Servos: enabled (Maestro connected)")
    elif SERVOS_ENABLED and not servo_ok:
        logger.warning("Servos: disabled (MAESTRO_PORT set but connection failed)")
    elif os.environ.get("DJR3X_NO_SERVOS", "").strip():
        logger.info("Servos: disabled (--noservos)")
    else:
        logger.info("Servos: disabled (MAESTRO_PORT not set)")

    head_ok = leds_head.connect()
    if HEAD_LEDS_ENABLED and head_ok:
        logger.info("Head LEDs: enabled (Arduino connected)")
    elif HEAD_LEDS_ENABLED and not head_ok:
        logger.warning("Head LEDs: disabled (ARDUINO_HEAD_PORT set but connection failed)")
    else:
        logger.info("Head LEDs: disabled (ARDUINO_HEAD_PORT not set)")

    chest_ok = leds_chest.connect()
    if CHEST_LEDS_ENABLED and chest_ok:
        logger.info("Chest LEDs: enabled (Arduino connected)")
    elif CHEST_LEDS_ENABLED and not chest_ok:
        logger.warning("Chest LEDs: disabled (ARDUINO_CHEST_PORT set but connection failed)")
    else:
        logger.info("Chest LEDs: disabled (ARDUINO_CHEST_PORT not set)")

    motion_master = bool(getattr(config, "MOTION_ENABLED", True))
    motion_ok = motion_controller.connect()
    if motion_ok:
        logger.info("Motion base: enabled (ESP32 connected)")
    elif not motion_master:
        logger.info("Motion base: disabled (config.MOTION_ENABLED is False)")
    elif MOTION_PORT_SET:
        logger.warning("Motion base: disabled (MOTION_ESP32_PORT set but connection/handshake failed)")
    else:
        logger.info("Motion base: disabled (MOTION_ESP32_PORT not set)")

    # Wire chest LEDs to state transitions so they stay in sync without
    # scattering leds_chest calls across every module.
    def _chest_state_callback(old: State, new: State) -> None:
        if new == State.ACTIVE:
            leds_chest.active()
        elif new == State.IDLE:
            leds_chest.idle()
        elif new == State.SLEEP:
            leds_chest.sleep()
        elif new == State.SHUTDOWN:
            # Don't fade here. The LED fade is fired in lockstep with the shutdown
            # audio and servo droop inside _shutdown() so the power-down reads as one
            # motion. Fading on the state flip ran the LEDs ~3-4s AHEAD of the sound
            # and servos (those wait for the main-loop poll + service teardown first).
            pass
    state.add_state_change_callback(_chest_state_callback)

    logger.info(
        "Camera: %s",
        f"enabled ({CAMERA_SELECTION_DESCRIPTION})" if CAMERA_ENABLED else "disabled",
    )
    logger.info(
        "Audio devices: %s",
        "disabled (--noaudio text-input mode)"
        if no_audio else (
            f"enabled ({AUDIO_SELECTION_DESCRIPTION})"
            if AUDIO_ENABLED else "disabled (AUDIO_DEVICE_NAME/AUDIO_DEVICE_INDEX not set or not found)"
        ),
    )

    # Step 5: animations module is ready — functions operate directly on the hardware
    # singletons initialized above. No AnimationPlayer class to instantiate.

    _abort_startup_if_shutdown("startup audio/animation")

    # Steps 6 & 7: Fire startup audio first, then run servo animation simultaneously.
    # Audio plays in a background thread so the servo motion begins immediately after.
    startup_audio_thread: threading.Thread | None = None
    if no_audio:
        logger.info("Startup audio disabled by --noaudio")
    elif config.PLAY_STARTUP_AUDIO:
        def _play_startup_audio() -> None:
            speech_choices = [
                str(c).strip()
                for c in (getattr(config, "STARTUP_SPEECH_CLIP_CHOICES", None) or [])
                if str(c).strip()
            ]
            choice_names = {Path(c).name.lower() for c in speech_choices}
            speech_state_path = str(getattr(config, "STARTUP_SPEECH_CLIP_STATE_PATH", "") or "")
            play_speech_clip = bool(getattr(config, "PLAY_STARTUP_SPEECH_CLIP", True))
            for audio_file in config.STARTUP_AUDIO_FILES:
                play_file = audio_file
                # The randomizable speech-clip slot: skip it entirely when the spoken
                # intro is disabled, otherwise swap in one of the choices (random,
                # never consecutive across launches).
                if speech_choices and Path(audio_file).name.lower() in choice_names:
                    if not play_speech_clip:
                        logger.info(
                            "Skipping startup speech clip %s (PLAY_STARTUP_SPEECH_CLIP=False)",
                            audio_file,
                        )
                        continue
                    picked = phrase_cycler.select_cycling_line(speech_choices, speech_state_path)
                    if picked:
                        play_file = picked
                logger.info("Playing startup audio: %s", play_file)
                try:
                    _play_audio_file(play_file)
                    if _is_listening_chime_audio_file(play_file):
                        speech_queue.mark_startup_chime_played()
                except Exception as e:
                    logger.warning("Could not play %s: %s", play_file, e)
        startup_audio_thread = threading.Thread(
            target=_play_startup_audio,
            daemon=True,
            name="startup_audio",
        )
        startup_audio_thread.start()
    else:
        logger.info("Startup audio disabled by config.PLAY_STARTUP_AUDIO")

    # Kick off the boot filler line NOW, chained behind the startup audio thread:
    # its worker pre-caches the TTS, waits for the last startup mp3 to finish, then
    # speaks after the configured beat — all while the blocking servo animation
    # below is still running. Starting it after the animation (the old order) left
    # ~8s of dead air between the spoken startup clip and the filler. Playback
    # needs no services: the mp3s already exercised the same audio output path.
    startup_boot_tts_thread = _start_startup_boot_tts_thread(wait_for=startup_audio_thread)

    logger.info("Playing startup animation...")
    animations.startup()
    if startup_audio_thread is not None and startup_audio_thread.is_alive():
        logger.info("Waiting for startup audio to finish before starting microphone services...")
        startup_audio_thread.join()

    _abort_startup_if_shutdown("background services")

    # Step 8: Start background services in order.
    logger.info("=== Starting background services ===")

    if no_audio:
        logger.info("Skipping audio.stream (--noaudio)")
    else:
        logger.info("Starting audio.stream...")
        stream.start()

    local_tts_mode = bool(getattr(config, "LOCAL_TTS_MODE", False))
    if no_audio:
        logger.info("Skipping audio output prewarm (--noaudio)")
    elif local_tts_mode:
        logger.info(
            "TTS: local on-device Qwen3-TTS (%s, voice=%s) — ElevenLabs disabled for this run",
            getattr(config, "LOCAL_TTS_MODEL_VARIANT", "?"),
            getattr(config, "LOCAL_TTS_VOICE", "?"),
        )
        logger.info("Pre-warming audio output device...")
        tts.prewarm()
        # No ElevenLabs warmup in local mode — the model is preloaded below instead.
    else:
        logger.info("TTS: ElevenLabs (Rex's true voice)")
        logger.info("Pre-warming audio output device...")
        tts.prewarm()
        # Open the ElevenLabs TLS connection in the background so the first live
        # reply doesn't pay the cold handshake (measured 2.7-5.6s first-turn TTS
        # outliers vs ~1.0-1.4s warm, 2026-07-06). Free metadata call, non-fatal.
        threading.Thread(target=tts.warmup_api, daemon=True, name="tts-api-warmup").start()

    # Arm the fine GIL switch interval EARLY — from the moment the boot filler
    # line can start playing (regression hardening 2026-07-18: the QoS used to
    # arm only at the preload block below, leaving the first line's opening
    # seconds unprotected if any background thread got busy). The PRE-QoS value
    # is captured HERE (not at the preload block, which would capture the fine
    # interval and make the "restore" a no-op).
    _prior_switch_interval = sys.getswitchinterval()
    if bool(getattr(config, "STARTUP_PRELOAD_AUDIO_QOS_ENABLED", True)) and not no_audio:
        sys.setswitchinterval(
            float(getattr(config, "STARTUP_PRELOAD_GIL_SWITCH_INTERVAL", 0.002))
        )

    # Kick off the boot line + head "look around" motion BEFORE the slow preloads
    # (Whisper / speaker-ID / Ollama) so "hang on folks while I'm booting up" plays
    # and the head scans the room WHILE the models load — instead of dead silence
    # and a frozen head, then the line once booting is basically already done.
    startup_scan_stop = threading.Event()
    startup_scan_thread: threading.Thread | None = None
    if (
        SERVOS_ENABLED
        and servo_ok
        and bool(getattr(config, "STARTUP_LOADING_SCAN_ENABLED", True))
    ):
        logger.info("Starting startup look-around scan during model loading...")
        startup_scan_thread = threading.Thread(
            target=animations.boot_scan_thread,
            args=(startup_scan_stop,),
            daemon=True,
            name="startup_loading_scan",
        )
        startup_scan_thread.start()

    # The boot filler line thread was started BEFORE the startup animation (chained
    # behind the startup audio clips) so it follows the spoken mp3 with no dead air;
    # by now it is typically already playing. Do NOT join it here: it is meant to
    # overlap the slow preloads below, covering the model-load wait. Its audio was
    # pre-cached in the worker so playback is a cheap cache hit, and the look-around
    # scan also keeps the head alive. We join it AFTER the preloads, below.

    # ── Audio QoS during the model preloads ──────────────────────────────────
    # The boot filler line plays WHILE the models load (that's the point of it).
    # But each load holds the GIL in long C-level bursts, which starves the
    # Python-level audio callback → the mid-sentence stutter. Two mitigations
    # while preloading (playback buffers are also deepened globally — see
    # config.AUDIO_PLAYBACK_BLOCKSIZE/LATENCY):
    #   1. a finer GIL switch interval so pure-Python import storms yield to the
    #      audio thread sooner;
    #   2. a short breath between load steps so the playback buffer refills
    #      after each burst (only while startup audio is actually playing).
    _audio_qos = (
        bool(getattr(config, "STARTUP_PRELOAD_AUDIO_QOS_ENABLED", True)) and not no_audio
    )
    if _audio_qos:
        # Interval already armed at the boot-line spawn above; just log the state.
        logger.info(
            "Audio QoS active for preloads (switch interval %.3fs, breath %.2fs)",
            sys.getswitchinterval(),
            float(getattr(config, "STARTUP_PRELOAD_BREATH_SECS", 0.25)),
        )

    def _preload_breath() -> None:
        """Give the audio callback room to refill its buffer after a load's GIL burst."""
        if not _audio_qos:
            return
        try:
            playing = output_gate.is_busy() or speech_queue.is_speaking() or tts.is_speaking()
        except Exception:
            playing = True
        if playing:
            time.sleep(float(getattr(config, "STARTUP_PRELOAD_BREATH_SECS", 0.25)))

    _abort_startup_if_shutdown("Whisper preload")
    if no_audio:
        logger.info("Skipping local Whisper preload (--noaudio)")
    elif bool(getattr(config, "WHISPER_PRELOAD_ON_STARTUP", True)):
        logger.info("Pre-loading local Whisper...")
        if not transcription.preload():
            logger.warning("Local Whisper preload failed; first transcription may be slower.")
    else:
        logger.info("Local Whisper preload disabled by config.WHISPER_PRELOAD_ON_STARTUP")
    _preload_breath()

    _abort_startup_if_shutdown("speaker ID preload")
    if no_audio:
        logger.info("Skipping speaker ID preload (--noaudio)")
    elif bool(getattr(config, "SPEAKER_ID_PRELOAD_ON_STARTUP", True)):
        logger.info("Pre-loading speaker ID encoder...")
        if not speaker_id.preload():
            logger.warning("Speaker ID encoder preload failed; continuing without voice ID.")
    else:
        logger.info("Speaker ID encoder preload disabled by config.SPEAKER_ID_PRELOAD_ON_STARTUP")
    _preload_breath()

    _abort_startup_if_shutdown("local LLM preload")
    logger.info("Pre-loading local LLM...")
    local_llm_ok = local_llm.preload()
    if not local_llm_ok and bool(getattr(config, "OLLAMA_PRELOAD_REQUIRED", True)):
        print(
            f"[FATAL] Local Ollama model could not be preloaded: {config.OLLAMA_MODEL}",
            file=sys.stderr,
        )
        print("Run:  python setup_assets.py", file=sys.stderr)
        sys.exit(1)
    if not local_llm_ok:
        logger.warning("Local LLM preload failed; continuing without local sidecar model.")
    _preload_breath()

    # ── Local Qwen3-TTS preload ──────────────────────────────────────────────
    # Preload the on-device voice when it's in use this run: --local-tts mode, or
    # LOCAL_TTS_WARM_ON_BOOT (so the first fallback line is instant). Otherwise the
    # ~2.9 GB model loads lazily on first use. This is NON-FATAL: ElevenLabs is
    # Rex's default and works, so a local-TTS problem logs a clear reason and Rex
    # degrades to ElevenLabs for the run (the per-line dispatch re-checks
    # availability, so a transient issue self-heals). Never blocks in --noaudio.
    if not no_audio:
        local_tts = None
        try:
            from audio import local_tts
        except Exception as exc:
            logger.error(
                "Local TTS engine import failed (%s) — Rex will use ElevenLabs.%s",
                exc, " Run: pip install -r requirements.txt" if local_tts_mode else "",
            )
        want_local = local_tts_mode or bool(getattr(config, "LOCAL_TTS_WARM_ON_BOOT", False))
        if local_tts is not None and want_local:
            _abort_startup_if_shutdown("local TTS preload")
            # require_rex_ref: speaking in Rex's LOCAL voice needs his reference
            # clip too — the dev-mac run had the model but not the (then-
            # gitignored) ref and silently fell back to ElevenLabs.
            reason = local_tts.unavailable_reason(require_rex_ref=True)
            if reason is not None:
                logger.error(
                    "Local TTS unavailable: %s.%s",
                    reason,
                    " Rex will use ElevenLabs for this run." if local_tts_mode
                    else " Fallback voice disabled.",
                )
            else:
                logger.info("Pre-loading local Qwen3-TTS voice model...")
                if local_tts.preload():
                    logger.info("Local Qwen3-TTS voice model ready.")
                else:
                    logger.error(
                        "Local TTS model failed to load (see the [local_tts] error above).%s",
                        " Rex will use ElevenLabs for this run." if local_tts_mode
                        else " Fallback voice disabled.",
                    )
            _preload_breath()

    if bool(getattr(config, "OPENAI_WARMUP_ON_STARTUP", True)):
        # Warm both OpenAI clients (answer LLM + action router) in the background
        # so the first turn doesn't eat cold TLS / connection setup. Non-blocking;
        # failures (e.g. offline / missing key) are swallowed inside warmup().
        def _warm_openai() -> None:
            try:
                from intelligence import llm, action_router
                llm.warmup()
                action_router.warmup()
            except Exception as exc:
                logger.debug("OpenAI warmup thread failed: %s", exc)

        threading.Thread(target=_warm_openai, daemon=True, name="openai-warmup").start()
        logger.info("OpenAI connection warmup started in background")

    _abort_startup_if_shutdown("animal detector preload")
    if bool(getattr(config, "LOCAL_ANIMAL_DETECTION_ENABLED", True)) and bool(
        getattr(config, "LOCAL_ANIMAL_DETECTION_PRELOAD_ON_STARTUP", True)
    ):
        logger.info("Pre-loading local animal detector...")
        if not animal_detector.preload():
            logger.warning(
                "Local animal detector preload failed; pet detection will retry during vision.scene."
            )
    else:
        logger.info("Local animal detector preload disabled by config.")

    # Preloads are done. Let the boot line finish (it played CONCURRENTLY with the loads
    # above, covering the wait) before the ready line / sensors take over, so nothing
    # talks over it. Then stop the look-around scan and recenter before sensors /
    # consciousness take the head, so face tracking inherits a known, centered pose.
    if startup_boot_tts_thread is not None and startup_boot_tts_thread.is_alive():
        startup_boot_tts_thread.join()
    # RF-DETR preload includes a costly first inference on its background thread.
    # Do not announce readiness while that work is still competing with PortAudio;
    # the 2026-07-22 live run stuttered throughout the ready line during this overlap.
    # The thinking effect started above fills this final wait.
    if bool(getattr(config, "LOCAL_ANIMAL_DETECTION_PRELOAD_ON_STARTUP", True)):
        if not animal_detector.wait_for_preload():
            logger.warning("Object detector preload did not finish cleanly before ready line.")
    # All CPU-heavy preload work is now complete; restore the normal GIL interval.
    if _audio_qos:
        sys.setswitchinterval(_prior_switch_interval)
    if startup_scan_thread is not None:
        startup_scan_stop.set()
        startup_scan_thread.join(timeout=3.0)

    _abort_startup_if_shutdown("sensor services")

    # audio.wake_word is started internally by intelligence.interaction.start() —
    # starting it separately would create a duplicate daemon thread.

    if no_audio:
        logger.info("Skipping audio.scene (--noaudio)")
    else:
        logger.info("Starting audio.scene...")
        audio_scene.start()

    camera.register_on_reconnect(_queue_camera_reconnect_line)

    logger.info("Starting vision.camera...")
    camera.start()

    logger.info("Starting vision.face_expression (MediaPipe local telemetry)...")
    face_expression.start()

    logger.info("Starting vision.pose (MediaPipe body pose / gesture)...")
    pose.start()

    logger.info("Starting vision.scene (periodic scan)...")
    vision_scene.start_periodic_scan(config.ENVIRONMENT_SCAN_INTERVAL_SECS)

    # Visual place recognition (perception/place_recognition.py). Loads MobileCLIP off the
    # main thread and publishes world_state.current_place; a no-op unless
    # PLACE_RECOGNITION_ENABLED and the encoder loads. Never blocks startup.
    logger.info("Starting perception.place_service (visual place recognition)...")
    place_service.start(emit_event=_place_event_sink)

    # Claim the listening chime now, BEFORE any startup speech can be enqueued.
    # main plays the single "ready" chime at the very end of startup (once all models
    # are loaded). Without this early claim, the first speech_queue.enqueue() during
    # startup (e.g. a sensor warning) makes the queue prepend its own listening chime —
    # so the user hears the chime twice (the prepended one, then main's ready chime).
    # mark_startup_chime_played() is idempotent, so the end-of-startup call is a no-op.
    if not no_audio and bool(getattr(config, "PLAY_LISTENING_CHIME", True)):
        speech_queue.mark_startup_chime_played()

    if not bool(
        getattr(config, "NO_AUDIO_MODE", False)
        or getattr(config, "AUDIO_OUTPUT_SUPPRESSED", False)
    ):
        camera_available = bool(CAMERA_ENABLED) and camera.wait_for_frame(
            float(getattr(config, "STARTUP_SENSOR_WARNING_CAMERA_WAIT_SECS", 2.5) or 0.0)
        )
        audio_available = bool(AUDIO_ENABLED) and stream.is_active()
        _queue_startup_device_warning(
            camera_available=camera_available,
            audio_available=audio_available,
        )

    logger.info("Starting awareness.chronoception...")
    chronoception.start_periodic_update()

    logger.info("Starting awareness.interoception...")
    interoception.start_periodic_update()

    # Resume Rex's preoccupation from last session (before consciousness/idle paths
    # can read it) so it carries across visits instead of re-rolling fresh.
    try:
        from intelligence import rex_pov
        if rex_pov.load_persisted():
            logger.info("Resumed Rex preoccupation: %s", rex_pov.active_seed_id())
    except Exception as exc:
        logger.debug("rex_pov load_persisted on startup failed: %s", exc)

    _abort_startup_if_shutdown("consciousness/interaction")
    logger.info("Starting intelligence.consciousness...")
    consciousness.start()

    if no_audio:
        logger.info("Starting intelligence.interaction (text-only; wake word disabled)...")
    else:
        logger.info("Starting intelligence.interaction (+ audio.wake_word)...")
    interaction.start(text_only=no_audio)

    # Step 9: Breathing thread and arm idle.
    logger.info("Starting servo breathing thread...")
    threading.Thread(
        target=servos.breathing_thread,
        daemon=True,
        name="breathing",
    ).start()

    logger.info("Starting servo wander thread...")
    threading.Thread(
        target=animations.wander_thread,
        daemon=True,
        name="wander",
    ).start()

    logger.info("Starting servo arm wander thread...")
    threading.Thread(
        target=animations.arm_wander_thread,
        daemon=True,
        name="arm_wander",
    ).start()

    logger.info("Setting arm to idle position...")
    animations.arm_idle()

    # Done-loading cue: now that all models are loaded and the wake word / VAD are
    # listening, play the chime so the user knows startup finished and Rex is ready.
    # (Moved here from the opening STARTUP_AUDIO_FILES burst.) The ownership claim that
    # suppresses the speech queue's duplicate listening chime is made earlier (before
    # the sensor-warning enqueue); the call below is a defensive, idempotent re-claim.
    # _play_audio_file is a no-op in --noaudio.
    _abort_startup_if_shutdown("ready line")
    # Startup warmup is over — end the looping "thinking" filler BEFORE announcing
    # readiness so the loop can't take another pass under the ready line. (It is
    # preemptible anyway; this just makes the handoff clean and immediate.)
    try:
        from audio import sound_effects as _sfx
        _sfx.stop_loop(_startup_thinking_loop)
    except Exception as exc:
        logger.debug("stopping startup thinking loop failed: %s", exc)
    if not no_audio and bool(getattr(config, "PLAY_LISTENING_CHIME", True)):
        ready_line = _select_startup_ready_tts_line()
        try:
            if ready_line:
                emotion = str(getattr(config, "STARTUP_READY_TTS_EMOTION", "neutral") or "neutral")
                logger.info("Speaking ready line — models loaded, listening: %s", ready_line)
                tts.speak(ready_line, emotion)
            else:
                # No ready lines configured → fall back to the original chime.
                logger.info("Playing ready chime — models loaded, listening.")
                _play_audio_file(config.LISTENING_CHIME_FILE)
            speech_queue.mark_startup_chime_played()
        except Exception as exc:
            logger.warning("Could not play ready signal: %s", exc)

    if startup_jeopardy:
        _abort_startup_if_shutdown("Jeopardy launch")
        _launch_startup_jeopardy()

    # Daily current-events fetch — deliberately AFTER the ready line (regression
    # fix 2026-07-18: fetching during the boot-filler window put client/TLS/JSON
    # work on the GIL exactly while the cached startup lines played -> stutter.
    # The stories are only consumed in conversation lulls; nothing needs them
    # before Rex is listening).
    try:
        from awareness import current_events
        current_events.start_background_refresh()
    except Exception as exc:
        logger.debug("current events refresh kick failed: %s", exc)

    # Compass fusion service (no-op until COMPASS_ENABLED).
    try:
        from hardware import compass as compass_service
        compass_service.start_service()
    except Exception as exc:
        logger.debug("compass service start failed: %s", exc)

    logger.info("=== DJ-R3X v2 is online ===")


def _wait_for_shutdown() -> None:
    """Keep-alive loop — exit when state transitions to SHUTDOWN."""
    try:
        while True:
            if state.is_state(State.SHUTDOWN):
                logger.info("SHUTDOWN state detected — beginning shutdown sequence.")
                break
            time.sleep(0.1)  # tight poll so the power-down starts right after the sign-off
    except KeyboardInterrupt:
        logger.info("KeyboardInterrupt received — initiating clean shutdown.")
        state.set_state(State.SHUTDOWN)


def _run_headless(*, startup_jeopardy: bool = False) -> None:
    try:
        _run_controller_startup(startup_jeopardy=startup_jeopardy)
        _wait_for_shutdown()
    except _StartupAborted:
        pass  # shutdown already requested; fall through to teardown
    except KeyboardInterrupt:
        logger.info("KeyboardInterrupt received during startup — initiating clean shutdown.")
        state.set_state(State.SHUTDOWN)
    finally:
        if state.is_state(State.SHUTDOWN):
            _shutdown()


def _gui_requested(args: argparse.Namespace) -> bool:
    return bool(getattr(args, "gui", False))


def _apply_cli_runtime_flags(args: argparse.Namespace) -> None:
    # Keep GUI-aware hardware mirrors/simulators off unless this launch asked
    # for the dashboard. config.py still owns the dashboard tuning knobs.
    setattr(config, "GUI_ENABLED", _gui_requested(args))
    no_audio = bool(getattr(args, "noaudio", False))
    setattr(config, "NO_AUDIO_MODE", no_audio)
    setattr(config, "AUDIO_OUTPUT_SUPPRESSED", no_audio)
    if no_audio:
        setattr(config, "PLAY_STARTUP_AUDIO", False)
        setattr(config, "PLAY_SHUTDOWN_AUDIO", False)
        setattr(config, "PLAY_LISTENING_CHIME", False)
        logger.info("--noaudio enabled: mic input, wake word, audio output, and ElevenLabs TTS are suppressed.")
    # --noservos took effect at import time (env seed -> config_loader SERVOS_ENABLED);
    # this is just the operator-visible confirmation.
    if bool(getattr(args, "noservos", False)):
        logger.info("--noservos enabled: Pololu Maestro servo control is disabled for this run.")


def _load_dashboard_runner():
    backend = str(getattr(config, "GUI_BACKEND", "pyside6") or "").strip().lower()
    if backend != "pyside6":
        logger.warning("GUI disabled: unsupported GUI_BACKEND=%r", backend)
        return None
    try:
        from gui.dashboard import run_dashboard
        return run_dashboard
    except Exception as exc:
        logger.warning("GUI disabled: could not import PySide6 dashboard: %s", exc)
        return None


def _start_gui_bridge_sync() -> None:
    """Copy camera/world state into the GUI bridge without giving Qt live objects."""
    global _gui_bridge_thread
    if _gui_bridge_thread and _gui_bridge_thread.is_alive():
        return

    _gui_bridge_stop.clear()

    def _sync_loop() -> None:
        try:
            from gui.state_bridge import gui_bridge
        except Exception as exc:
            logger.warning("GUI bridge sync unavailable: %s", exc)
            return

        fps = max(1, int(getattr(config, "GUI_FPS", 20) or 20))
        interval = 1.0 / float(fps)
        while not _gui_bridge_stop.is_set():
            try:
                if bool(getattr(config, "GUI_CAMERA_PREVIEW_ENABLED", True)):
                    gui_bridge.update_frame(camera.get_frame())
                else:
                    gui_bridge.update_frame(None)
            except Exception:
                pass
            try:
                gui_bridge.update_camera_stats(**camera.frame_info())
            except Exception:
                pass
            try:
                from world_state import world_state as _ws
                snapshot = _ws.snapshot()
                snapshot["state"] = state.get_state().value
                people = list(snapshot.get("people") or [])
                if people:
                    enriched_people = []
                    for person in people:
                        person = dict(person)
                        person_id = person.get("person_db_id")
                        local_expression = person.get("face_expression")
                        local_mood = person.get("face_mood")
                        has_local_expression = (
                            isinstance(local_expression, dict)
                            and local_expression.get("source") == "mediapipe_face_landmarker"
                        ) or (
                            isinstance(local_mood, dict)
                            and local_mood.get("source") == "mediapipe_face_landmarker"
                        )
                        if person_id is not None:
                            try:
                                mood = consciousness.get_cached_mood(int(person_id))
                            except Exception:
                                mood = None
                            if mood and not has_local_expression:
                                person["face_mood"] = mood
                                expression = (
                                    str(person.get("expression") or "").strip().lower()
                                )
                                if expression in {"", "neutral", "unknown", "none"}:
                                    person["expression"] = mood.get("mood") or "neutral"
                        enriched_people.append(person)
                    snapshot["people"] = enriched_people
                gui_bridge.update_world_state_snapshot(snapshot)
            except Exception as exc:
                logger.debug("GUI bridge world-state sync failed: %s", exc)
            try:
                from features import games as games_mod
                gui_bridge.update_game_state_snapshot(games_mod.snapshot())
            except Exception as exc:
                logger.debug("GUI bridge game-state sync failed: %s", exc)
            try:
                gui_bridge.update_speech_state(
                    speaking=speech_queue.is_speaking(),
                    audio_path=speech_queue.current_audio_path(),
                )
            except Exception as exc:
                logger.debug("GUI bridge speech-state sync failed: %s", exc)
            _gui_bridge_stop.wait(interval)

    _gui_bridge_thread = threading.Thread(
        target=_sync_loop,
        daemon=True,
        name="gui-bridge-sync",
    )
    _gui_bridge_thread.start()


def _stop_gui_bridge_sync() -> None:
    _gui_bridge_stop.set()
    if _gui_bridge_thread and _gui_bridge_thread.is_alive():
        _gui_bridge_thread.join(timeout=2.0)


def _make_gui_text_submit_callback(ready=None):
    def _submit(text: str) -> None:
        cleaned = (text or "").strip()
        if not cleaned:
            return

        # The dashboard now opens before interaction.start() — don't drive the
        # turn pipeline while services are still booting.
        if ready is not None and not ready():
            try:
                from utils import conv_log
                conv_log.log_system(
                    "Rex is still booting — the status turns Connected when he's ready."
                )
            except Exception:
                pass
            return

        def _worker() -> None:
            try:
                handled = interaction.submit_text(cleaned)
                if not handled:
                    from utils import conv_log
                    import config as _config
                    if getattr(_config, "INTERACTION_PAUSED", False):
                        conv_log.log_system(
                            "Rex is paused (Memory Banks editor open) — close it to chat."
                        )
                    else:
                        conv_log.log_system(
                            "Text input was ignored because Rex is asleep or shutting down."
                        )
            except Exception as exc:
                logger.exception("GUI text input failed: %s", exc)
                try:
                    from utils import conv_log
                    conv_log.log_system(f"Text input failed: {exc}")
                except Exception:
                    pass

        threading.Thread(
            target=_worker,
            daemon=True,
            name="gui-text-input",
        ).start()

    return _submit


def _install_gui_log_mirror() -> None:
    """Mirror app-log records into the GUI bridge for the dashboard's system-log
    panel. Installed before anything else in GUI mode so the panel captures the
    whole startup sequence (the dashboard now opens while startup is running)."""
    try:
        from gui.state_bridge import gui_bridge
        from utils.logging import install_gui_log_handler

        install_gui_log_handler(gui_bridge.add_log_line)
    except Exception as exc:
        logger.debug("GUI log mirror unavailable: %s", exc)


def _run_gui_mode(run_dashboard, *, startup_jeopardy: bool = False) -> None:
    """GUI-first startup: show the dashboard immediately on the main thread (Qt
    must own it on macOS) and run the heavy controller startup — hardware, audio,
    model preloads, services — on a background thread. The window opens within a
    couple of seconds instead of after the full boot; the top-bar status shows
    'Booting…' until startup completes and the system-log panel streams progress."""
    _install_gui_log_mirror()

    startup_done = threading.Event()
    controller_online = threading.Event()
    startup_exit: dict[str, int] = {}

    def _set_controller_status(status: str) -> None:
        try:
            from gui.state_bridge import gui_bridge
            gui_bridge.update_controller_status(status)
        except Exception:
            pass

    if startup_jeopardy:
        # Tell the dashboard up front so it opens on the Jeopardy page during
        # boot instead of showing the diagnostic dashboard to the audience.
        try:
            from gui.state_bridge import gui_bridge
            gui_bridge.set_startup_game_intent("jeopardy")
        except Exception:
            pass

    def _startup_worker() -> None:
        try:
            _run_controller_startup(startup_jeopardy=startup_jeopardy)
            _set_controller_status("online")
            controller_online.set()
        except _StartupAborted as exc:
            # User closed the window / hit Ctrl+C mid-boot; not a failure.
            logger.info("Controller startup stopped at checkpoint: %s", exc)
        except SystemExit as exc:
            # The fatal startup paths call sys.exit(); in a thread that only
            # raises SystemExit here — surface it and bring the whole app down.
            # Mark "failed" BEFORE SHUTDOWN: the dashboard keeps the window open
            # on failed status so the system-log panel stays readable.
            code = exc.code if isinstance(exc.code, int) else 1
            startup_exit["code"] = code or 1
            logger.critical(
                "Controller startup aborted (exit %s) — see the system log.", exc.code
            )
            _set_controller_status("failed")
            state.set_state(State.SHUTDOWN)
        except Exception:
            startup_exit["code"] = 1
            logger.exception("Controller startup failed — see the system log.")
            _set_controller_status("failed")
            state.set_state(State.SHUTDOWN)
        finally:
            startup_done.set()

    startup_thread = threading.Thread(
        target=_startup_worker,
        daemon=True,
        name="controller-startup",
    )

    try:
        _start_gui_bridge_sync()
        startup_thread.start()

        def _request_shutdown() -> None:
            state.set_state(State.SHUTDOWN)

        # Sleep/wake block on speech, so run them off the Qt thread. Guarded so a
        # stray click can't double-fire. These are the same entry points the spoken
        # "go to sleep" command and the wake word use.
        def _request_sleep() -> None:
            if state.is_state(State.SLEEP):
                return

            def _run() -> None:
                try:
                    interaction._enter_sleep_mode()
                except Exception as exc:
                    logger.warning("GUI sleep request failed: %s", exc)

            threading.Thread(target=_run, daemon=True, name="gui-sleep").start()

        def _request_wake() -> None:
            if not state.is_state(State.SLEEP):
                return
            if bool(getattr(config, "SLEEP_ONNX_ONLY_WAKE", True)):
                logger.info("GUI wake ignored — SLEEP requires the wakeuprex ONNX model")
                return

            def _run() -> None:
                try:
                    interaction._wake_from_sleep()
                except Exception as exc:
                    logger.warning("GUI wake request failed: %s", exc)

            threading.Thread(target=_run, daemon=True, name="gui-wake").start()

        try:
            run_dashboard(
                shutdown_callback=_request_shutdown,
                text_submit_callback=_make_gui_text_submit_callback(
                    ready=controller_online.is_set
                ),
                sleep_callback=_request_sleep,
                wake_callback=_request_wake,
            )
        except Exception as exc:
            logger.warning("GUI failed at runtime; continuing headless: %s", exc)
            _wait_for_shutdown()
    except KeyboardInterrupt:
        logger.info("KeyboardInterrupt received in GUI mode — initiating clean shutdown.")
        state.set_state(State.SHUTDOWN)
    finally:
        state.set_state(State.SHUTDOWN)
        # run_dashboard restored the default SIGINT handler on exit, so a second
        # Ctrl+C here would raise INSIDE this finally block and abandon the rest
        # of it — skipping _shutdown() (servo droop, LED off, session save).
        # Teardown is already underway; absorb further Ctrl+C instead.
        try:
            signal.signal(
                signal.SIGINT,
                lambda *_: logger.info("Shutdown already in progress..."),
            )
        except (ValueError, OSError):
            pass
        # Let an in-flight startup finish before tearing services down —
        # _shutdown() stopping modules mid-start is the racier path. The abort
        # checkpoints in _run_controller_startup stop the boot at the next phase
        # boundary, so this normally resolves in seconds; the bound is a backstop
        # so a hung preload can never wedge shutdown.
        if startup_thread.is_alive():
            logger.info("Waiting for controller startup to stop before shutdown...")
            if not startup_done.wait(timeout=120.0):
                logger.warning("Controller startup still running after 120s — shutting down anyway.")
        if startup_exit.get("code"):
            # Failed boot: skip the power-down theatrics, just tear down quietly.
            setattr(config, "PLAY_SHUTDOWN_AUDIO", False)
        _stop_gui_bridge_sync()
        _shutdown()
        if startup_exit.get("code"):
            sys.exit(startup_exit["code"])


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run the DJ-R3X controller.")
    parser.add_argument(
        "-jeopardy",
        "--jeopardy",
        action="store_true",
        help="skip startup introductions and go directly into the Jeopardy game",
    )
    parser.add_argument(
        "-gui",
        "--gui",
        action="store_true",
        help="open the optional PySide6 GUI dashboard for this run",
    )
    parser.add_argument(
        "-noaudio",
        "--noaudio",
        "--no-audio",
        dest="noaudio",
        action="store_true",
        help="disable microphone capture and audio output; use GUI text input/output instead",
    )
    parser.add_argument(
        "-noservos",
        "--noservos",
        "--no-servos",
        dest="noservos",
        action="store_true",
        help="disable the Pololu Maestro servo controller entirely for this run",
    )
    parser.add_argument(
        "-local-tts",
        "--local-tts",
        "--localtts",
        dest="local_tts",
        action="store_true",
        help="use the on-device Qwen3-TTS voice clone instead of ElevenLabs for this run",
    )
    return parser.parse_args(argv)


def _acquire_single_instance_or_exit() -> None:
    """Take the cross-process single-instance lock, or exit if a controller is
    already running.

    The always-on supervisor (rex_supervisor.py) launches main.py on "wake up
    rex". This guard makes a duplicate launch harmless: if a controller is
    already alive — even just asleep — the second process logs and exits 0
    instead of fighting over the mic/servos. The lock auto-frees when the holder
    dies, so a crash never strands it. Set DJR3X_SKIP_SINGLE_INSTANCE=1 to bypass
    (manual dev runs, tests).
    """
    if os.environ.get("DJR3X_SKIP_SINGLE_INSTANCE", "").strip() in ("1", "true", "True"):
        return
    try:
        from utils import single_instance
    except Exception as exc:  # never let the guard itself prevent startup
        logger.warning("Single-instance lock unavailable (%s) — continuing.", exc)
        return
    if single_instance.acquire():
        logger.info("Single-instance lock acquired (%s).", single_instance.lock_path())
        return
    owner = single_instance.read_owner_pid()
    logger.warning(
        "DJ-R3X controller is already running (pid=%s, lock=%s). Exiting this "
        "duplicate launch.",
        owner if owner is not None else "unknown",
        single_instance.lock_path(),
    )
    # Clean exit code so the supervisor/launchd treats it as a no-op, not a crash.
    sys.exit(0)


def main(argv: list[str] | None = None) -> None:
    args = _parse_args(argv)
    _acquire_single_instance_or_exit()
    _apply_cli_runtime_flags(args)
    if not _gui_requested(args):
        _run_headless(startup_jeopardy=args.jeopardy)
        return

    run_dashboard = _load_dashboard_runner()
    if run_dashboard is None:
        _run_headless(startup_jeopardy=args.jeopardy)
        return

    _run_gui_mode(run_dashboard, startup_jeopardy=args.jeopardy)


if __name__ == "__main__":
    main()
