#!/usr/bin/env python3
"""
tools/test_active_speaker.py — Active-speaker detection calibration / diagnostic.

Runs the live camera through the visual active-speaker pipeline and logs, per
cycle, the per-face head yaw + lip-motion energy and the chosen speaker — so the
energy/margin/yaw thresholds in config.py can be tuned on-device. Mirrors
tools/test_voice_id.py in spirit (no full Rex stack required).

It seeds world_state.people with one anonymous slot per detected face (face_visible
= True), so the SAME IoU association the production hook uses can run without the
dlib identity stack. Speaker identities therefore show as slot:N here.

VAD source (which gates whether anyone is the active speaker):
    --vad force   speech always considered active — calibrate lip motion alone (default)
    --vad mic     run the real Silero VAD on the mic (needed for the chew/yawn test)
    --vad off     never active — sanity check that nothing is ever selected

Usage:
    python tools/test_active_speaker.py                 # 30s, VAD forced on
    python tools/test_active_speaker.py --secs 60 --vad mic
    python tools/test_active_speaker.py --landmark-yaw  # also log the fallback yaw

Calibration scenarios (docs/active_speaker_detection.md §10): single talker;
two alternating; distractors (chew/yawn/laugh, --vad mic); turn-away; leave/return.
"""

import argparse
import logging
import os
import sys
import threading
import time
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_ROOT))
os.chdir(_ROOT)

try:
    from dotenv import load_dotenv
    load_dotenv(_ROOT / ".env")
except Exception:
    pass

import config
# Turn on the per-cycle scoreboard log BEFORE the module reads the flag.
config.ACTIVE_SPEAKER_LOG_SCOREBOARD = True
config.ACTIVE_SPEAKER_ENABLED = True

from world_state import world_state
from vision import camera
from vision import face_expression
from vision import active_speaker
from awareness.situation import assessor


def _setup_logging() -> None:
    h = logging.StreamHandler(sys.stdout)
    h.setFormatter(logging.Formatter("%(asctime)s | %(message)s", "%H:%M:%S"))
    root = logging.getLogger("vision.active_speaker")
    root.setLevel(logging.INFO)
    root.addHandler(h)
    root.propagate = False


def _seed_slots_for(expressions) -> None:
    """Give each detected face an anonymous, visible world_state slot so the
    production IoU association (_match_expression_to_people) can run without the
    dlib identity pipeline."""
    people = []
    for i, expr in enumerate(expressions):
        people.append({
            "id": f"person_{i + 1}",
            "person_db_id": None,
            "face_id": None,
            "face_box": expr.get("face_box"),
            "face_visible": True,
            "face_missing": False,
        })
    world_state.update("people", people)


def _mic_vad_thread(stop_event: threading.Event) -> None:
    """Feed real Silero VAD from the mic into the assessor (mirrors interaction.py)."""
    try:
        import numpy as np
        import sounddevice as sd
        from audio import vad as vad_mod
        from utils.config_loader import AUDIO_DEVICE_INDEX
    except Exception as exc:
        print(f"  mic VAD unavailable ({exc}); falling back to forced VAD.")
        while not stop_event.is_set():
            assessor.set_vad_active(True)
            stop_event.wait(0.1)
        return

    sr = config.AUDIO_SAMPLE_RATE
    block = int(sr * 0.1)
    with sd.InputStream(samplerate=sr, channels=1, dtype="float32",
                        device=AUDIO_DEVICE_INDEX, blocksize=block) as stream:
        while not stop_event.is_set():
            chunk, _ = stream.read(block)
            mono = chunk[:, 0] if chunk.ndim > 1 else chunk
            try:
                speech = bool(vad_mod.is_speech(mono))
            except Exception:
                speech = True
            assessor.set_vad_active(speech)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--secs", type=float, default=30.0, help="How long to run (default 30)")
    ap.add_argument("--vad", choices=["force", "mic", "off"], default="force")
    ap.add_argument("--landmark-yaw", action="store_true",
                    help="Also log the landmark-asymmetry fallback yaw for comparison")
    args = ap.parse_args()

    _setup_logging()
    print(f"Active-speaker calibration — {args.secs:.0f}s, VAD={args.vad}")
    print(f"  thresholds: LIPSYNC_ENERGY={config.LIPSYNC_ENERGY_THRESHOLD} "
          f"FACING_YAW_MAX_DEG={config.FACING_YAW_MAX_DEG} "
          f"SPEAKER_MARGIN={config.SPEAKER_MARGIN}")

    if not face_expression._load_model():
        print("ERROR: MediaPipe Face Landmarker failed to load (run setup_assets.py).")
        return 1

    camera.start()
    if not camera.wait_for_frame(timeout_secs=3.0):
        print("ERROR: no camera frames (check CAMERA_INDEX/CAMERA_DEVICE_NAME in .env).")
        camera.stop()
        return 1

    stop_event = threading.Event()
    vad_thread = None
    if args.vad == "mic":
        vad_thread = threading.Thread(target=_mic_vad_thread, args=(stop_event,), daemon=True)
        vad_thread.start()
    elif args.vad == "off":
        assessor.set_vad_active(False)

    interval = float(getattr(config, "FACE_EXPRESSION_ANALYSIS_INTERVAL_SECS", 0.25))
    deadline = time.time() + args.secs
    try:
        while time.time() < deadline:
            if args.vad == "force":
                assessor.set_vad_active(True)
            frame = camera.get_frame()
            if frame is None:
                time.sleep(interval)
                continue
            expressions = face_expression.detect_expressions(frame)
            _seed_slots_for(expressions)
            matches: list = []
            face_expression.merge_expressions_into_world_state(expressions, collect_matches=matches)
            by_expr = {ei: (si, pid) for (ei, si, pid) in matches}
            signals = []
            now = time.time()
            for ei, expr in enumerate(expressions):
                if ei not in by_expr:
                    continue
                si, pid = by_expr[ei]
                yaw = active_speaker.yaw_from_transform_matrix(expr.get("transform_matrix"))
                if args.landmark_yaw:
                    lyaw = active_speaker._yaw_from_landmarks(expr.get("landmarks"))
                    print(f"    slot:{si} matrix_yaw={yaw} landmark_yaw={lyaw} jaw={expr.get('jaw_open'):.3f}")
                signals.append({"slot_idx": si, "person_db_id": pid,
                                "jaw_open": float(expr.get("jaw_open") or 0.0),
                                "yaw": yaw, "ts": now})
            active_speaker.update(signals, vad_active=assessor.is_user_speaking())
            time.sleep(interval)
    except KeyboardInterrupt:
        print("\n  interrupted.")
    finally:
        stop_event.set()
        if vad_thread is not None:
            vad_thread.join(timeout=1.0)
        camera.stop()
        active_speaker.reset()
    print("done.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
