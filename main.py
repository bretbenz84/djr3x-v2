#!/usr/bin/env python3
"""DJ-R3X v2 — main entry point."""

import sys
from pathlib import Path

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

import pygame

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
)
from sequences import animations
from audio import stream, scene as audio_scene, output_gate, tts
from vision import camera, scene as vision_scene
from awareness import chronoception, interoception
from intelligence import consciousness, interaction


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
    """Play a pre-recorded audio file synchronously via pygame."""
    with output_gate.hold("startup_or_shutdown_clip") as acquired:
        if not acquired:
            return
        pygame.mixer.music.load(path)
        pygame.mixer.music.play()
        while pygame.mixer.music.get_busy():
            time.sleep(0.05)


def _shutdown() -> None:
    logger.info("=== Shutdown sequence begin ===")

    # Stop services in reverse startup order.
    logger.info("Stopping intelligence.interaction...")
    interaction.stop()  # also calls wake_word.stop() internally

    logger.info("Stopping intelligence.consciousness...")
    consciousness.stop()

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
    # Join the audio thread before pygame teardown so the clip isn't cut short.
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

    pygame.mixer.quit()
    logger.info("=== Shutdown complete ===")


def main() -> None:
    logger.info("Verifying local Whisper model...")
    _verify_local_whisper_model()

    pygame.mixer.init()

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
    logger.info("Audio devices: %s", "enabled" if AUDIO_ENABLED else "disabled (AUDIO_DEVICE_INDEX not set)")

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

    logger.info("Setting arm to idle position...")
    animations.arm_idle()

    logger.info("=== DJ-R3X v2 is online ===")

    # Step 10: Keep-alive loop — exit when state transitions to SHUTDOWN.
    try:
        while True:
            if state.is_state(State.SHUTDOWN):
                logger.info("SHUTDOWN state detected — beginning shutdown sequence.")
                break
            time.sleep(1.0)
    except KeyboardInterrupt:
        logger.info("KeyboardInterrupt received — initiating clean shutdown.")
        state.set_state(State.SHUTDOWN)

    # Step 11: Shutdown.
    _shutdown()


if __name__ == "__main__":
    main()
