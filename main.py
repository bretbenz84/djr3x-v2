#!/usr/bin/env python3
"""DJ-R3X v2 — main entry point."""

import os
import sys
import argparse
from pathlib import Path

_PROJECT_ROOT = Path(__file__).resolve().parent
_PROJECT_VENV = (_PROJECT_ROOT / "venv").resolve()


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

# Step 3: Load config — raises RuntimeError at import time if API keys are missing.
logger.info("Loading configuration and API keys...")
try:
    from utils import config_loader  # noqa: F401  triggers import-time key validation
except RuntimeError as e:
    logger.critical("Configuration error — missing or invalid API key: %s", e)
    sys.exit(1)

# All remaining imports after config is confirmed valid.
import time
import threading

import numpy as np
import sounddevice as sd
import soundfile as sf

import config
import state
from state import State
from hardware import servos, leds_head, leds_chest
from utils.config_loader import (
    SERVOS_ENABLED,
    HEAD_LEDS_ENABLED,
    CHEST_LEDS_ENABLED,
    CAMERA_ENABLED,
    CAMERA_SELECTION_DESCRIPTION,
    AUDIO_ENABLED,
    AUDIO_SELECTION_DESCRIPTION,
)
from sequences import animations
from audio import stream, scene as audio_scene, output_gate, tts, speech_queue
from vision import camera, scene as vision_scene
from awareness import chronoception, interoception
from intelligence import consciousness, interaction, local_llm


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


def _play_audio_file(path: str) -> None:
    """Play a pre-recorded audio file synchronously via sounddevice.

    Using sounddevice (PortAudio) here — not pygame.mixer (SDL) — keeps all
    playback on a single backend. Concurrent SDL + PortAudio on macOS produces
    PaMacCore err=-50 glitches, which surface as choppy clip playback and
    duplicated/echoed TTS output.
    """
    with output_gate.hold("startup_or_shutdown_clip") as acquired:
        if not acquired:
            return
        audio, samplerate = sf.read(path, dtype="float32", always_2d=False)
        if audio.ndim > 1:
            audio = audio.mean(axis=1).astype(np.float32)
        sd.play(audio, samplerate, blocksize=2048)
        sd.wait()


def _play_listening_chime_async(reason: str) -> None:
    """Queue the listening chime through speech_queue so AEC suppresses it."""
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


def _shutdown() -> None:
    logger.info("=== Shutdown sequence begin ===")

    # Stop services in reverse startup order.
    logger.info("Stopping intelligence.interaction...")
    interaction.stop()  # also calls wake_word.stop() internally

    logger.info("Stopping intelligence.consciousness...")
    consciousness.stop()

    logger.info("Unloading local LLM...")
    local_llm.unload()

    logger.info("Stopping awareness.interoception...")
    interoception.stop()

    logger.info("Stopping awareness.chronoception...")
    chronoception.stop()

    logger.info("Stopping vision.scene...")
    vision_scene.stop()

    logger.info("Stopping vision.camera...")
    camera.stop()

    logger.info("Stopping audio.scene...")
    audio_scene.stop()

    logger.info("Stopping audio.stream...")
    stream.stop()

    # Fire shutdown audio first, then run servo animation simultaneously.
    # Join the audio thread so the clip isn't cut short by shutdown teardown.
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
    animations.shutdown()

    if _audio_thread is not None:
        _audio_thread.join()

    # Close hardware.
    logger.info("Closing hardware...")
    servos.shutdown()
    leds_head.disconnect()
    leds_chest.disconnect()

    logger.info("=== Shutdown complete ===")


_gui_bridge_stop = threading.Event()
_gui_bridge_thread: threading.Thread | None = None


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
    logger.info("Verifying local Whisper model...")
    _verify_local_whisper_model()

    # Step 4: Initialize hardware and log enabled/disabled status.
    logger.info("=== Initializing hardware ===")

    servo_ok = servos.connect()
    if SERVOS_ENABLED and servo_ok:
        logger.info("Servos: enabled (Maestro connected)")
    elif SERVOS_ENABLED and not servo_ok:
        logger.warning("Servos: disabled (MAESTRO_PORT set but connection failed)")
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
            leds_chest.off()
    state.add_state_change_callback(_chest_state_callback)

    logger.info(
        "Camera: %s",
        f"enabled ({CAMERA_SELECTION_DESCRIPTION})" if CAMERA_ENABLED else "disabled",
    )
    logger.info(
        "Audio devices: %s",
        f"enabled ({AUDIO_SELECTION_DESCRIPTION})"
        if AUDIO_ENABLED else "disabled (AUDIO_DEVICE_NAME/AUDIO_DEVICE_INDEX not set or not found)",
    )

    # Step 5: animations module is ready — functions operate directly on the hardware
    # singletons initialized above. No AnimationPlayer class to instantiate.

    # Steps 6 & 7: Fire startup audio first, then run servo animation simultaneously.
    # Audio plays in a background thread so the servo motion begins immediately after.
    if config.PLAY_STARTUP_AUDIO:
        def _play_startup_audio() -> None:
            for audio_file in config.STARTUP_AUDIO_FILES:
                logger.info("Playing startup audio: %s", audio_file)
                try:
                    _play_audio_file(audio_file)
                except Exception as e:
                    logger.warning("Could not play %s: %s", audio_file, e)
        threading.Thread(target=_play_startup_audio, daemon=True, name="startup_audio").start()
    else:
        logger.info("Startup audio disabled by config.PLAY_STARTUP_AUDIO")

    logger.info("Playing startup animation...")
    animations.startup()

    # Step 8: Start background services in order.
    logger.info("=== Starting background services ===")

    logger.info("Starting audio.stream...")
    stream.start()

    logger.info("Pre-warming audio output device...")
    tts.prewarm()

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

    # audio.wake_word is started internally by intelligence.interaction.start() —
    # starting it separately would create a duplicate daemon thread.

    logger.info("Starting audio.scene...")
    audio_scene.start()

    logger.info("Starting vision.camera...")
    camera.start()

    logger.info("Starting vision.scene (periodic scan)...")
    vision_scene.start_periodic_scan(config.ENVIRONMENT_SCAN_INTERVAL_SECS)

    logger.info("Starting awareness.chronoception...")
    chronoception.start_periodic_update()

    logger.info("Starting awareness.interoception...")
    interoception.start_periodic_update()

    logger.info("Starting intelligence.consciousness...")
    consciousness.start()

    logger.info("Starting intelligence.interaction (+ audio.wake_word)...")
    interaction.start()

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

    if startup_jeopardy:
        _launch_startup_jeopardy()

    logger.info("=== DJ-R3X v2 is online ===")


def _wait_for_shutdown() -> None:
    """Keep-alive loop — exit when state transitions to SHUTDOWN."""
    try:
        while True:
            if state.is_state(State.SHUTDOWN):
                logger.info("SHUTDOWN state detected — beginning shutdown sequence.")
                break
            time.sleep(1.0)
    except KeyboardInterrupt:
        logger.info("KeyboardInterrupt received — initiating clean shutdown.")
        state.set_state(State.SHUTDOWN)


def _run_headless(*, startup_jeopardy: bool = False) -> None:
    _run_controller_startup(startup_jeopardy=startup_jeopardy)
    _wait_for_shutdown()
    _shutdown()


def _gui_requested() -> bool:
    return bool(getattr(config, "GUI_ENABLED", False))


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
                from world_state import world_state as _ws
                snapshot = _ws.snapshot()
                snapshot["state"] = state.get_state().value
                people = list(snapshot.get("people") or [])
                if people:
                    enriched_people = []
                    for person in people:
                        person = dict(person)
                        person_id = person.get("person_db_id")
                        if person_id is not None and not any(
                            person.get(key)
                            for key in ("expression", "mood", "emotion", "affect", "face_mood")
                        ):
                            try:
                                mood = consciousness.get_cached_mood(int(person_id))
                            except Exception:
                                mood = None
                            if mood:
                                person["face_mood"] = mood
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


def _run_gui_mode(run_dashboard, *, startup_jeopardy: bool = False) -> None:
    _run_controller_startup(startup_jeopardy=startup_jeopardy)
    _start_gui_bridge_sync()

    shutdown_started = threading.Event()

    def _request_shutdown() -> None:
        state.set_state(State.SHUTDOWN)

    def _shutdown_watcher() -> None:
        _wait_for_shutdown()
        if not shutdown_started.is_set():
            shutdown_started.set()
            _shutdown()

    watcher = threading.Thread(
        target=_shutdown_watcher,
        daemon=False,
        name="shutdown-watcher",
    )
    watcher.start()

    try:
        try:
            run_dashboard(shutdown_callback=_request_shutdown)
        except Exception as exc:
            logger.warning("GUI failed at runtime; continuing headless: %s", exc)
            _wait_for_shutdown()
    except KeyboardInterrupt:
        logger.info("KeyboardInterrupt received in GUI mode — initiating clean shutdown.")
        state.set_state(State.SHUTDOWN)
    finally:
        state.set_state(State.SHUTDOWN)
        watcher.join(timeout=60.0)
        if watcher.is_alive():
            logger.warning("Shutdown watcher did not finish within timeout")
        _stop_gui_bridge_sync()


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run the DJ-R3X controller.")
    parser.add_argument(
        "-jeopardy",
        "--jeopardy",
        action="store_true",
        help="skip startup introductions and go directly into the Jeopardy game",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    args = _parse_args(argv)
    if not _gui_requested():
        _run_headless(startup_jeopardy=args.jeopardy)
        return

    run_dashboard = _load_dashboard_runner()
    if run_dashboard is None:
        _run_headless(startup_jeopardy=args.jeopardy)
        return

    _run_gui_mode(run_dashboard, startup_jeopardy=args.jeopardy)


if __name__ == "__main__":
    main()
