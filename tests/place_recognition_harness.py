#!/usr/bin/env python3
"""
tests/place_recognition_harness.py — offline CLI for perception.place_recognition.

Enroll rooms from labeled image directories and query others, printing a per-room
accuracy table, a confusion summary, and the score distribution — so the PLACE_MATCH_*
thresholds can be tuned against real house imagery WITHOUT the robot.

Directory layout (one subdirectory per room, images inside):
    dataset/office/IMG_0001.jpg ...
    dataset/living_room/*.jpg ...

Subcommands:
    enroll --db PATH --images DIR [--model-tag TAG] [--embedder {mobileclip,mock}]
           [--no-gates] [--clip-model NAME] [--clip-pretrained TAG] [--clip-checkpoint PATH]
    query  --db PATH --images DIR [--model-tag TAG] [--embedder {mobileclip,mock}] [...]

This drives the REAL module code paths — ``PlaceRecognizer.enroll_from_frames`` and
``PlaceRecognizer.score_frame`` — with an injected image embedder. Scoring is never
reimplemented here.

Embedders:
    mobileclip (default) — a real MobileCLIP-S2 image encoder loaded via ``open_clip``
        (``pip install open_clip_torch``) or Apple's ``mobileclip`` package (pass
        ``--clip-checkpoint``). This is the encoder the on-robot vision stack uses.
    mock — a deterministic coarse color-gist embedding (numpy + PIL only). For plumbing
        / CI verification of the pipeline ONLY; it is NOT MobileCLIP and its accuracy
        numbers do not reflect real place recognition.
"""

from __future__ import annotations

import argparse
import os
import sys
from collections import Counter, defaultdict

import numpy as np

# Make the project root importable (config.py + perception/ live there).
_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

from perception.place_recognition import (  # noqa: E402
    CONFIDENT, TENTATIVE, UNKNOWN, PlaceRecognizer,
)

_IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tif", ".tiff"}


# ── Image loading ────────────────────────────────────────────────────────────────

def _load_rgb(path: str) -> np.ndarray:
    """Load an image as an RGB uint8 HxWx3 ndarray (identical representation for enroll
    and query — the module and embedder see exactly the same thing either way)."""
    try:
        from PIL import Image
        with Image.open(path) as im:
            return np.asarray(im.convert("RGB"), dtype=np.uint8)
    except ImportError:
        import cv2  # mediapipe pulls in opencv, so this is available on-robot
        bgr = cv2.imread(path, cv2.IMREAD_COLOR)
        if bgr is None:
            raise IOError(f"could not read image: {path}")
        return cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)


def _list_rooms(images_dir: str):
    """Yield (room_name, [image_paths]) for each subdirectory holding images."""
    if not os.path.isdir(images_dir):
        raise SystemExit(f"not a directory: {images_dir}")
    for room in sorted(os.listdir(images_dir)):
        sub = os.path.join(images_dir, room)
        if not os.path.isdir(sub):
            continue
        paths = [
            os.path.join(sub, f)
            for f in sorted(os.listdir(sub))
            if os.path.splitext(f)[1].lower() in _IMAGE_EXTS
        ]
        if paths:
            yield room, paths


# ── Embedders ─────────────────────────────────────────────────────────────────────

def _mock_embed_fn():
    """Deterministic coarse color-gist embedding (8x8x3 = 192-d). Plumbing/CI only."""
    def embed(frame: np.ndarray) -> np.ndarray:
        from PIL import Image
        im = Image.fromarray(np.asarray(frame, dtype=np.uint8)).convert("RGB").resize((8, 8))
        return np.asarray(im, dtype=np.float32).reshape(-1)
    return embed


def _mobileclip_embed_fn(model_name: str, pretrained: str, checkpoint: str | None):
    """Real MobileCLIP-S2 image encoder. Tries open_clip, then Apple's mobileclip pkg."""
    errors = []
    try:
        import torch
        import open_clip
        model, _, preprocess = open_clip.create_model_and_transforms(
            model_name, pretrained=(checkpoint or pretrained)
        )
        model.eval()

        def embed(frame: np.ndarray) -> np.ndarray:
            from PIL import Image
            img = Image.fromarray(np.asarray(frame, dtype=np.uint8)).convert("RGB")
            with torch.no_grad():
                feats = model.encode_image(preprocess(img).unsqueeze(0))
            return feats[0].float().cpu().numpy()

        return embed
    except Exception as exc:  # noqa: BLE001
        errors.append(f"open_clip: {exc!r}")

    try:
        import torch
        import mobileclip
        mc_name = model_name.lower().replace("-", "_")  # "MobileCLIP-S2" -> "mobileclip_s2"
        model, _, preprocess = mobileclip.create_model_and_transforms(
            mc_name, pretrained=checkpoint
        )
        model.eval()

        def embed(frame: np.ndarray) -> np.ndarray:
            from PIL import Image
            img = Image.fromarray(np.asarray(frame, dtype=np.uint8)).convert("RGB")
            with torch.no_grad():
                feats = model.encode_image(preprocess(img).unsqueeze(0))
            return feats[0].float().cpu().numpy()

        return embed
    except Exception as exc:  # noqa: BLE001
        errors.append(f"mobileclip: {exc!r}")

    raise SystemExit(
        "Could not load a MobileCLIP encoder. Install one of:\n"
        "  pip install open_clip_torch      (then --embedder mobileclip works out of the box)\n"
        "  pip install mobileclip           (then pass --clip-checkpoint /path/to/mobileclip_s2.pt)\n"
        "or use --embedder mock for a plumbing-only run.\n"
        "Errors:\n  " + "\n  ".join(errors)
    )


def _build_embed_fn(args):
    if args.embedder == "mock":
        return _mock_embed_fn()
    return _mobileclip_embed_fn(args.clip_model, args.clip_pretrained, args.clip_checkpoint)


def _make_recognizer(args):
    return PlaceRecognizer(
        _build_embed_fn(args),
        db_path=args.db,
        model_tag=args.model_tag,
    )


# ── Subcommands ─────────────────────────────────────────────────────────────────

def cmd_enroll(args) -> None:
    pr = _make_recognizer(args)
    try:
        total = 0
        for room, paths in _list_rooms(args.images):
            frames = [_load_rgb(p) for p in paths]
            res = pr.enroll_from_frames(room, frames, run_duplicate_check=False)
            note = "" if res.committed == res.provided else f" (capped from {res.provided})"
            print(f"  enrolled {res.name:<22} {res.committed:>3} embeddings{note}")
            total += res.committed
        print(f"\nEnrolled {total} embeddings across {len(pr.place_names())} places "
              f"into {args.db} (tag={args.model_tag}).")
    finally:
        pr.close()


def cmd_query(args) -> None:
    pr = _make_recognizer(args)
    try:
        rooms = list(_list_rooms(args.images))
        if not rooms:
            raise SystemExit(f"no room subdirectories with images under {args.images}")
        known = pr.place_names()
        if not known:
            raise SystemExit(f"{args.db} has no enrolled places for tag {args.model_tag}; "
                             f"run the enroll subcommand first")

        confusion = defaultdict(Counter)          # true_room -> Counter(predicted)
        per_room = defaultdict(lambda: [0, 0])    # true_room -> [correct, total]
        cls_counts = Counter()
        best_scores, correct_scores, wrong_scores = [], [], []

        for true_room, paths in rooms:
            for p in paths:
                res = pr.score_frame(_load_rgb(p))
                predicted = res.best.name if res.best else None
                label = predicted if res.classification != UNKNOWN else "<unknown>"
                confusion[true_room][label] += 1
                cls_counts[res.classification] += 1
                per_room[true_room][1] += 1
                is_correct = predicted == true_room and res.classification != UNKNOWN
                if is_correct:
                    per_room[true_room][0] += 1
                if res.best is not None:
                    best_scores.append(res.best.score)
                    (correct_scores if is_correct else wrong_scores).append(res.best.score)

        _print_report(known, confusion, per_room, cls_counts,
                      best_scores, correct_scores, wrong_scores)
    finally:
        pr.close()


# ── Reporting ─────────────────────────────────────────────────────────────────────

def _print_report(known, confusion, per_room, cls_counts,
                  best_scores, correct_scores, wrong_scores) -> None:
    total = sum(t for _, t in per_room.values())
    correct = sum(c for c, _ in per_room.values())

    print("\n=== Per-room accuracy ===")
    print(f"  {'room':<22} {'correct':>8} {'total':>6} {'acc':>7}")
    for room in sorted(per_room):
        c, t = per_room[room]
        acc = (c / t) if t else 0.0
        print(f"  {room:<22} {c:>8} {t:>6} {acc:>6.1%}")
    print(f"  {'-'*22} {'-'*8} {'-'*6} {'-'*7}")
    overall = (correct / total) if total else 0.0
    print(f"  {'OVERALL':<22} {correct:>8} {total:>6} {overall:>6.1%}")

    print("\n=== Confusion (rows = true room, cols = predicted) ===")
    cols = known + ["<unknown>"]
    corner = "true \\ pred"
    header = "  " + f"{corner:<18}" + "".join(f"{_abbrev(c):>10}" for c in cols)
    print(header)
    for room in sorted(confusion):
        row = confusion[room]
        cells = "".join(f"{row.get(c, 0):>10}" for c in cols)
        print(f"  {_abbrev(room, 18):<18}{cells}")

    print("\n=== Classification mix ===")
    for cls in (CONFIDENT, TENTATIVE, UNKNOWN):
        n = cls_counts.get(cls, 0)
        frac = (n / total) if total else 0.0
        print(f"  {cls:<10} {n:>5}  {frac:>6.1%}")

    print("\n=== Best-score distribution ===")
    _print_hist("all best scores", best_scores)
    _print_hist("correct matches", correct_scores)
    _print_hist("wrong/unknown  ", wrong_scores)


def _print_hist(label: str, scores, bins=(0.0, 0.5, 0.6, 0.68, 0.72, 0.76, 0.80, 0.85, 0.9, 1.0)) -> None:
    if not scores:
        print(f"  {label}: (none)")
        return
    arr = np.asarray(scores, dtype=np.float32)
    counts, _ = np.histogram(arr, bins=list(bins))
    parts = " ".join(f"[{bins[i]:.2f},{bins[i+1]:.2f}):{counts[i]}" for i in range(len(counts)))
    print(f"  {label}: n={len(arr)} mean={arr.mean():.3f} min={arr.min():.3f} "
          f"max={arr.max():.3f}\n      {parts}")


def _abbrev(s: str, width: int = 9) -> str:
    return s if len(s) <= width else s[: width - 1] + "…"


# ── CLI ───────────────────────────────────────────────────────────────────────────

def _add_common(sp) -> None:
    sp.add_argument("--db", required=True, help="path to places.db (created if missing)")
    sp.add_argument("--images", required=True, help="dir of per-room image subdirectories")
    sp.add_argument("--model-tag", default="mobileclip_s2_v1",
                    help="model_tag written/filtered on (default: mobileclip_s2_v1)")
    sp.add_argument("--embedder", choices=("mobileclip", "mock"), default="mobileclip",
                    help="image encoder (default: mobileclip; mock = plumbing only)")
    sp.add_argument("--clip-model", default="MobileCLIP-S2", help="open_clip model name")
    sp.add_argument("--clip-pretrained", default="datacompdr", help="open_clip pretrained tag")
    sp.add_argument("--clip-checkpoint", default=None,
                    help="checkpoint path for Apple's mobileclip package")


def main(argv=None) -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)

    e = sub.add_parser("enroll", help="enroll each subdir as a place")
    _add_common(e)
    e.add_argument("--no-gates", action="store_true", default=True,
                   help="bypass heading/time diversity gates (always on for bulk enroll)")
    e.set_defaults(func=cmd_enroll)

    q = sub.add_parser("query", help="score images and print accuracy/confusion")
    _add_common(q)
    q.set_defaults(func=cmd_query)

    args = ap.parse_args(argv)
    args.func(args)


if __name__ == "__main__":
    main()
