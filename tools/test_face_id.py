#!/usr/bin/env python3
"""
tools/test_face_id.py — Face-recognition diagnostic + enrollment.

Grabs a frame from the camera, runs the active face backend (InsightFace by
default), and prints a ranked distance scoreboard against every enrolled face
in the DB — or enrolls/re-enrolls a person's face.

Needed after the dlib → InsightFace migration: 128-dim dlib enrollments are
skipped by the 512-dim ArcFace matcher, so known people must re-enroll their
face once (voice is unaffected). Stale rows are harmless; --replace clears them.

Usage:
    python tools/test_face_id.py                      # detect + scoreboard
    python tools/test_face_id.py --enroll "Name"      # add a face biometric
    python tools/test_face_id.py --enroll "Name" --replace
                                                      # replace ALL prior face rows with a fresh one
    python tools/test_face_id.py --image path.jpg     # use an image file instead of the camera

No Rex stack required. Uses the project's config + DB + camera.
"""

import argparse
import os
import sys
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

import numpy as np

import config
from memory import database as db
from memory import people
from vision import face


def _grab_frame(image_path: str | None):
    if image_path:
        import cv2
        frame = cv2.imread(image_path)
        if frame is None:
            sys.exit(f"could not read image: {image_path}")
        return frame

    # Use the project's camera module (handles the ffmpeg AVFoundation wrapper).
    from vision import camera
    camera.start()
    if not camera.wait_for_frame(timeout_secs=5.0):
        camera.stop()
        sys.exit("camera not available — pass --image path.jpg instead")
    time.sleep(0.5)  # let exposure settle
    frame = camera.get_frame()
    camera.stop()
    if frame is None:
        sys.exit("camera read failed")
    return frame


def _scoreboard(encoding: np.ndarray) -> None:
    rows = db.fetchall("SELECT person_id, encoding FROM biometrics WHERE type = 'face'")
    query = encoding.astype(np.float32)
    scored: dict[int, float] = {}
    skipped = 0
    for row in rows:
        stored = np.frombuffer(bytes(row["encoding"]), dtype=np.float32)
        if stored.shape != query.shape:
            skipped += 1
            continue
        d = float(np.linalg.norm(stored - query))
        pid = row["person_id"]
        scored[pid] = min(d, scored.get(pid, float("inf")))

    if query.shape[-1] == 512:
        thr = float(getattr(config, "FACE_RECOGNITION_DISTANCE_THRESHOLD_ARCFACE", 1.10))
    else:
        thr = float(config.FACE_RECOGNITION_DISTANCE_THRESHOLD)

    print(f"\n  Distance scoreboard (backend={face.active_backend()}, "
          f"dim={query.shape[-1]}, accept < {thr:.2f}):")
    if not scored:
        print("    (no comparable enrolled faces)")
    for pid, d in sorted(scored.items(), key=lambda kv: kv[1]):
        person = people.get_person(pid) or {}
        name = person.get("name") or f"person_{pid}"
        verdict = "MATCH" if d < thr else "no"
        print(f"    {name:<24} d={d:.3f}  {verdict}")
    if skipped:
        print(f"    ({skipped} stale other-backend row(s) skipped — re-enroll those people)")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[1].strip())
    ap.add_argument("--enroll", metavar="NAME", help="store the largest detected face for NAME")
    ap.add_argument("--replace", action="store_true",
                    help="with --enroll: delete ALL prior face rows for NAME first")
    ap.add_argument("--image", help="use an image file instead of the camera")
    args = ap.parse_args()

    frame = _grab_frame(args.image)
    print(f"frame: {frame.shape[1]}x{frame.shape[0]}")

    dets = face.detect_faces(frame)
    print(f"backend={face.active_backend()}  faces detected: {len(dets)}")
    if not dets:
        sys.exit(1)

    for d in dets:
        x, y, w, h = d["bounding_box"]
        conf = d.get("confidence")
        conf_s = f" conf={conf:.2f}" if conf is not None else ""
        print(f"  face at ({x},{y}) {w}x{h}{conf_s}")

    largest = max(dets, key=lambda f: f["bounding_box"][2] * f["bounding_box"][3])

    if args.enroll:
        row = db.fetchone("SELECT id FROM people WHERE name = ? COLLATE NOCASE", (args.enroll,))
        if row is None:
            sys.exit(f"no person named {args.enroll!r} in people.db")
        pid = row["id"]
        if args.replace:
            db.execute("DELETE FROM biometrics WHERE person_id = ? AND type = 'face'", (pid,))
            print(f"  deleted prior face rows for {args.enroll}")
        result = people.add_biometric(pid, "face", largest["encoding"])
        if result is None:
            sys.exit("  DB write failed")
        print(f"  enrolled {args.enroll} (person_id={pid}, dim={largest['encoding'].shape[-1]})")

    _scoreboard(largest["encoding"])


if __name__ == "__main__":
    main()
