#!/usr/bin/env python3
"""Run relevant unittest modules in isolated processes with real I/O blocked."""
import os
from pathlib import Path
import subprocess
import sys
import tempfile

ROOT = Path(__file__).resolve().parents[1]
MODULES = (
    "campplus voice_backend voice_signatures speaker_id_margin identity_instrumentation passive_voice_enroll "
    "conversation_arc lean_context_state speech_generations delivery_contract "
    "optional_local_work turn_coordinator runtime_report gap_speech "
    "action_results_and_attribution memory_semantic memory_unified_retrieval "
    "semantic_breaker_and_motion_vocab streaming_tts two_chunk_tts turn_trace "
    "addressee lean_impulse_menu production_replay restructure_ownership llm_compat "
    "dialogue_act turn_plan turn_planner_slim_contract conversation_streaming "
    "voice_primary_identity voice_bearing_match active_speaker lean_multi_party "
    "game_roster_identity motion_swing motion_sequence motion_route_tool speaker_segments camping_identity_regression"
).split()


def child(module):
    os.environ["DJR3X_NO_SERVOS"] = "1"
    sys.path.insert(0, str(ROOT))
    sys.modules["mlx"] = None
    sys.modules["mlx_whisper"] = None
    def forbidden(*args, **kwargs):
        raise RuntimeError("real hardware/network access forbidden in lean checks")
    import socket
    socket.socket.connect = forbidden
    import serial
    serial.Serial = forbidden
    import sounddevice
    for name in ("play", "InputStream", "OutputStream", "RawInputStream", "RawOutputStream"):
        setattr(sounddevice, name, forbidden)
    with tempfile.TemporaryDirectory(prefix="rex-check-") as temp:
        import config
        config.DB_PATH = str(Path(temp) / "people.db")
        config.REX_DB_PATH = str(Path(temp) / "rex.db")
        config.PLACE_DB_PATH = str(Path(temp) / "places.db")
        config.TTS_CACHE_DIR = str(Path(temp) / "tts")
        import sqlite3
        from setup_assets import DB_SCHEMA
        with sqlite3.connect(config.DB_PATH) as db:
            db.executescript(DB_SCHEMA)
        import unittest
        sys.argv = ["unittest", "tests.test_" + module]
        unittest.main(module=None)


def main():
    if len(sys.argv) == 3 and sys.argv[1] == "--child":
        child(sys.argv[2])
        return
    failures = []
    for module in sys.argv[1:] or MODULES:
        try:
            run = subprocess.run([sys.executable, __file__, "--child", module], cwd=ROOT,
                                 text=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, timeout=60)
            summary = " | ".join(line for line in run.stdout.splitlines()
                                 if line.startswith(("Ran ", "OK", "FAILED")))
            print(module, run.returncode, summary, flush=True)
            if run.returncode:
                failures.append(module)
                print(run.stdout[-12000:], flush=True)
        except subprocess.TimeoutExpired:
            failures.append(module)
            print(module, "TIMEOUT", flush=True)
    print("Failed modules:", failures, flush=True)
    raise SystemExit(bool(failures))


if __name__ == "__main__":
    main()
