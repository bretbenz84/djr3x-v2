#!/usr/bin/env python3
"""
tools/voice_face_test.py — does the voice come from where the face is?

One take: records a few seconds of you talking (the robot's mic path, channel
1 × AUDIO_INPUT_GAIN), polls the Flex XVF3800's direction of arrival for the
same window, grabs a camera frame mid-utterance, then reports side by side:

  - the voice bearing (dominant DoA cluster, base frame, + = left)
  - every face in the frame: who it is (face DB), its bearing from the box
    position + the neck yaw, and how far it sits from the voice
  - the voice-ID scoreboard for the recording
  - the verdict the live attribution code would reach
    (perception.voice_bearing_match — the same function the robot uses)

    ./venv/bin/python tools/voice_face_test.py                 # 6 s take, neck centred
    ./venv/bin/python tools/voice_face_test.py --secs 8 --neck-deg 0
    ./venv/bin/python tools/voice_face_test.py --label "bret 20deg right"
    ./venv/bin/python tools/voice_face_test.py --fit      # calibrate the lens from every take so far

Run it YOURSELF from a terminal (it waits for Enter, then counts down, then
records — start talking at "RECORDING"). Rex must be stopped (camera + mic).
--neck-deg is the head's yaw off the body, + = Rex's right; leave the head
centred and the default 0 stands. The annotated frame is saved under
logs/mic_check/ and the numbers are appended to logs/mic_check/voice_face.jsonl.

Records the mic and reads the camera; makes no sound and moves nothing.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import threading
import time
from datetime import datetime
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_ROOT))
os.chdir(_ROOT)

try:
    from dotenv import load_dotenv
    load_dotenv(_ROOT / ".env")
except Exception:
    pass

import numpy as np  # noqa: E402

import config  # noqa: E402
from perception import voice_bearing_match as vbm  # noqa: E402


def _countdown(msg: str, secs: int = 3) -> None:
    print(f"\n{msg}")
    for i in range(secs, 0, -1):
        print(f"   starting in {i}...", end="\r", flush=True)
        time.sleep(1)
    print("   RECORDING — talk now      ")


def _lens_kwargs() -> dict:
    return {
        "px_per_deg": float(getattr(config, "VOICE_BEARING_CAM_PX_PER_DEG", 0.0) or 0.0),
        "yaw_offset_deg": float(getattr(config, "VOICE_BEARING_CAM_YAW_OFFSET_DEG", 0.0) or 0.0),
    }


def _fit() -> int:
    """Calibrate the camera's angular model against the voice bearing using every
    logged take with exactly one face. The voice is the reference (±3° on a
    steady talker), the face box gives pixels — the fit returns px/deg and the
    constant yaw offset. Do takes at the frame CENTRE and near BOTH EDGES."""
    path = _ROOT / "logs" / "mic_check" / "voice_face.jsonl"
    if not path.exists():
        print("no takes logged yet")
        return 1
    samples = []
    print(f"{'ts':>15} {'label':<18} {'px':>6} {'voice':>7} {'neck':>5}")
    for line in path.read_text().splitlines():
        try:
            r = json.loads(line)
        except Exception:
            continue
        voice = r.get("voice") or {}
        faces = r.get("faces") or []
        if not voice or len(faces) != 1 or voice.get("bearing_deg") is None:
            continue
        x, _y, w, _h = faces[0]["box"]
        width = float(getattr(config, "CAMERA_WIDTH", 1920) or 1920)
        px = (float(x) + float(w) / 2.0) - width / 2.0
        neck = float(r.get("neck_deg") or 0.0)
        samples.append((px, float(voice["bearing_deg"]), neck))
        print(f"{r['ts']:>15} {str(r.get('label') or '')[:18]:<18} {px:6.0f} {float(voice['bearing_deg']):+7.1f} {neck:+5.0f}")
    fit = vbm.fit_camera_model(samples)
    if fit is None:
        print("\nnot enough spread to fit (need ≥2 one-face takes at different frame positions)")
        return 1
    print(f"\n  fit over {fit['n']} takes: {fit['px_per_deg']:.1f} px/deg "
          f"(half-frame ≈ {960.0 / fit['px_per_deg']:.0f}°), yaw offset {fit['yaw_offset_deg']:+.1f}° "
          f"(+ = camera axis right of the mic's 0°), rms residual {fit['rms_deg']:.1f}°")
    print("  residuals per take (deg): " + ", ".join(f"{r:+.1f}" for r in fit["residuals"]))
    if fit["n"] < 4:
        print("  ! fewer than 4 takes — add one at the frame centre and one near each edge before trusting this")
    print("\n  to adopt: in config.py set")
    print(f"    VOICE_BEARING_CAM_PX_PER_DEG = {fit['px_per_deg']:.1f}")
    print(f"    VOICE_BEARING_CAM_YAW_OFFSET_DEG = {fit['yaw_offset_deg']:.1f}   # ONLY if the head was centred in every take;")
    print("                                             # otherwise this is the neck yaw the live app reads itself")
    return 0


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--secs", type=float, default=6.0, help="recording length")
    ap.add_argument("--neck-deg", type=float, default=0.0,
                    help="head yaw off the body during the take, + = Rex's right (default 0 = centred)")
    ap.add_argument("--frame-at", type=float, default=1.5, help="seconds into the take to grab the frame")
    ap.add_argument("--label", default="", help="tag for the log record")
    ap.add_argument("--fit", action="store_true",
                    help="no recording: fit px-per-degree + yaw offset from logs/mic_check/voice_face.jsonl")
    args = ap.parse_args()
    if args.fit:
        return _fit()

    from tools import mic_check
    from hardware import flex_doa
    from vision import camera, face
    from audio import speaker_id
    from memory import people

    print("─" * 68)
    print("Voice ↔ face bearing test")
    print(f"  take {args.secs:.0f}s, frame at {args.frame_at:.1f}s, neck yaw {args.neck_deg:+.0f}° (+ = right)")
    lens = _lens_kwargs()
    print(f"  lens model: {lens['px_per_deg']:.1f} px/deg, yaw offset {lens['yaw_offset_deg']:+.1f}°"
          if lens["px_per_deg"] > 0 else "  lens model: fraction × half-FOV (uncalibrated)")
    print(f"  mic channel {getattr(config, 'AUDIO_AEC_INPUT_CHANNEL', -1)}, gain "
          f"{getattr(config, 'AUDIO_INPUT_GAIN', 1.0)}x, camera half-FOV "
          f"{getattr(config, 'MOTION_COME_CAM_HALF_FOV_DEG', 25.0):.0f}°, frame width "
          f"{getattr(config, 'CAMERA_WIDTH', 1920)}")
    print("─" * 68)

    print("loading face + voice models...")
    face_ok = face._load_models()
    voice_ok = speaker_id.preload()
    print(f"  face backend: {face.active_backend() if face_ok else 'unavailable'}   "
          f"voice backend: {speaker_id.active_backend() if voice_ok else 'unavailable'}")

    print("starting camera...")
    camera.start()
    if not camera.wait_for_frame(timeout_secs=6.0):
        camera.stop()
        print("camera not available — is Rex still running?")
        return 1
    if not flex_doa.start():
        camera.stop()
        print("Flex DoA poller would not start (is the Flex the configured mic?)")
        return 1
    for _ in range(30):
        if flex_doa.available():
            break
        time.sleep(0.1)
    if not flex_doa.available():
        print(f"DoA poller not connected: {flex_doa.status()}")
        camera.stop()
        flex_doa.stop()
        return 1

    input("\nSit/stand where you want to be tested, face the camera, then press Enter... ")
    _countdown(f"Recording {args.secs:.0f}s — speak the whole time", 3)

    grabbed: dict = {}

    def _grab():
        time.sleep(max(0.0, args.frame_at))
        grabbed["frame"] = camera.get_frame()
        grabbed["at"] = time.monotonic()

    t0 = time.monotonic()
    threading.Thread(target=_grab, daemon=True).start()
    raw = mic_check._record(args.secs)
    t1 = time.monotonic()
    mono = mic_check._as_pipeline_mono(raw)
    time.sleep(0.2)
    voice = flex_doa.bearing_between(t0, t1)
    camera.stop()
    flex_doa.stop()

    frame = grabbed.get("frame")
    if frame is None:
        frame = camera.get_frame()
    print(f"\n  audio: {mic_check._dbfs(mono):.1f} dBFS RMS, voiced {speaker_id.voiced_secs(mono):.1f}s")

    # ── voice bearing ────────────────────────────────────────────────────────
    if voice is None:
        print("  voice bearing: NONE (chip never flagged speech, or no dominant cluster)")
    else:
        print(f"  voice bearing: {voice['bearing_deg']:+.0f}° base frame (chip {voice['raw_deg']:.0f}°), "
              f"{voice['cluster_n']}/{voice['n']} samples agree, spread {voice['spread_deg']:.0f}°")

    # ── faces ────────────────────────────────────────────────────────────────
    people_rows = []
    dets = face.detect_faces(frame) if frame is not None else []
    names: dict = {}
    for d in dets:
        x, y, w, h = d["bounding_box"]
        match = None
        try:
            match = face.identify_face(d["encoding"])
        except Exception as exc:
            print(f"  face id failed: {exc}")
        pid = int(match["id"]) if match else None
        name = (match or {}).get("name") or "unknown"
        names[pid] = name
        people_rows.append({"person_db_id": pid, "face_box": (int(x), int(y), int(w), int(h)),
                            "face_id": name})
    width = float(frame.shape[1]) if frame is not None else float(getattr(config, "CAMERA_WIDTH", 1920))
    half_fov = float(getattr(config, "MOTION_COME_CAM_HALF_FOV_DEG", 25.0))
    print(f"\n  faces in frame: {len(dets)}")
    for p in people_rows:
        b = vbm.face_bearing_deg(p, args.neck_deg, frame_width=width, half_fov_deg=half_fov, **_lens_kwargs())
        x, y, w, h = p["face_box"]
        px = (x + w / 2.0) - width / 2.0
        print(f"    {p['face_id']:<20} box ({x},{y}) {w}x{h}  {px:+.0f} px off centre  bearing {b:+.0f}° base frame")

    # ── voice id ─────────────────────────────────────────────────────────────
    ranked = []
    try:
        ranked = speaker_id.rank_speakers(mono)[:3]
    except Exception as exc:
        print(f"  voice id failed: {exc}")
    print("\n  voice ID scoreboard:")
    for pid, name, score, n_prints in ranked:
        print(f"    {name:<20} {score:.3f}  ({n_prints} prints)")
    if not ranked:
        print("    (nothing scored)")

    # ── the live matcher ─────────────────────────────────────────────────────
    result = None
    if voice is not None and people_rows:
        result = vbm.match_faces_to_voice(
            people_rows, voice["bearing_deg"], args.neck_deg,
            frame_width=width, half_fov_deg=half_fov,
            tolerance_deg=float(getattr(config, "VOICE_BEARING_FACE_TOLERANCE_DEG", 20.0)),
            margin_deg=float(getattr(config, "VOICE_BEARING_FACE_MARGIN_DEG", 10.0)),
            contradiction_deg=float(getattr(config, "VOICE_BEARING_CONTRADICTION_DEG", 45.0)),
            **_lens_kwargs(),
        )
        print(f"\n  matcher: {vbm.describe(result, names)}")
        top_voice_pid = ranked[0][0] if ranked else None
        confirm = result.get("confirm_pid")
        if confirm is not None and top_voice_pid is not None and int(confirm) == int(top_voice_pid):
            print(f"  VERDICT: consistent — the voice bearing lands on {names.get(confirm)}'s face and the "
                  f"voice print agrees ({ranked[0][2]:.3f}).")
        elif confirm is not None:
            print(f"  VERDICT: bearing says {names.get(confirm)}, voice print says "
                  f"{ranked[0][1] if ranked else '?'} — check enrolments.")
        elif result.get("contradicts"):
            print("  VERDICT: the voice came from where NO face is — off-camera talker.")
        else:
            print("  VERDICT: inconclusive — nearest face is outside the tolerance but not far enough "
                  "to call it off-camera. Compare the two bearings above; if they differ by a "
                  "constant, that is the neck yaw or FLEX_DOA_FORWARD_OFFSET_DEG.")
    elif voice is not None:
        print("\n  no face detected — nothing to match the voice against")

    # ── save ─────────────────────────────────────────────────────────────────
    outdir = _ROOT / "logs" / "mic_check"
    outdir.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    if frame is not None:
        try:
            import cv2
            img = frame.copy()
            for p in people_rows:
                x, y, w, h = p["face_box"]
                b = vbm.face_bearing_deg(p, args.neck_deg, frame_width=width, half_fov_deg=half_fov, **_lens_kwargs())
                cv2.rectangle(img, (x, y), (x + w, y + h), (0, 255, 0), 2)
                cv2.putText(img, f"{p['face_id']} {b:+.0f}", (x, max(20, y - 8)),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            if voice is not None:
                cv2.putText(img, f"voice {voice['bearing_deg']:+.0f} deg", (20, 40),
                            cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 200, 255), 2)
            path = outdir / f"voice_face-{stamp}.jpg"
            cv2.imwrite(str(path), img)
            print(f"\n  frame saved: {path}")
        except Exception as exc:
            print(f"  (frame not saved: {exc})")
    record = {
        "ts": stamp, "label": args.label, "neck_deg": args.neck_deg,
        "voice": voice,
        "faces": [{"pid": p["person_db_id"], "name": p["face_id"], "box": p["face_box"],
                   "bearing_deg": vbm.face_bearing_deg(p, args.neck_deg, frame_width=width, half_fov_deg=half_fov, **_lens_kwargs())}
                  for p in people_rows],
        "voice_id": [{"pid": pid, "name": name, "score": score} for pid, name, score, _ in ranked],
        "match": (None if result is None else
                  {k: v for k, v in result.items() if k != "faces"} | {
                      "faces": [{"pid": r["pid"], "bearing_deg": r["bearing_deg"], "delta_deg": r["delta_deg"]}
                                for r in result["faces"]]}),
    }
    with (outdir / "voice_face.jsonl").open("a") as f:
        f.write(json.dumps(record, default=float) + "\n")
    print(f"  logged to {outdir / 'voice_face.jsonl'}")
    return 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    except KeyboardInterrupt:
        print("\ninterrupted")
