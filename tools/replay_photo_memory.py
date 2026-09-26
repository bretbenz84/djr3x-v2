"""Opt-in live API sanity check using existing album images; never operate Rex.

This is a minimum recognition check, not a held-out accuracy benchmark. It
tests both reference orders and abstention with the correct identity removed.
Images never leave the machine unless --live is supplied.
"""
import argparse
import json
import os
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--live", action="store_true", help="Send saved photos to the configured OpenAI API")
    parser.add_argument("--model", help="Evaluate a model without changing production configuration")
    args = parser.parse_args()
    if not args.live:
        print("No requests sent. --live uploads saved photos and uses API credits.")
        return
    os.environ["DJR3X_NO_SERVOS"] = "1"
    # Keep API access; explicitly disable physical I/O even if imports change.
    def forbidden(*args, **kwargs):
        raise RuntimeError("Hardware is forbidden during photo replay")
    import serial
    import sounddevice
    serial.Serial = forbidden
    for name in ("play", "InputStream", "OutputStream", "RawInputStream", "RawOutputStream"):
        setattr(sounddevice, name, forbidden)
    sys.modules["mlx"] = None
    sys.modules["mlx_whisper"] = None
    import config
    from memory import photo_album
    from vision import photo_memory
    if args.model:
        config.PHOTO_MEMORY_MODEL = args.model
    import logging
    photo_memory._log.setLevel(logging.INFO)
    photo_memory._log.addHandler(logging.StreamHandler())
    refs = photo_album.references("animal", "dog")
    if len(refs) != 2:
        raise SystemExit("This bounded replay requires exactly two pet identities (six requests).")
    cases = []
    for ref in refs:
        for variant, gallery in (
            ("forward", refs), ("reverse", list(reversed(refs))),
            ("other-only", [r for r in refs if r["id"] != ref["id"]]),
        ):
            expected = None if variant == "other-only" else ref["name"]
            error = None
            try:
                result = photo_memory._compare(
                    {"status": "unknown", "jpeg": ref["images"][0], "label": "dog", "kind": "animal"},
                    gallery)
            except Exception as exc:
                error = type(exc).__name__
                result = {"status": "error"}
            actual = result.get("name")
            row = {"subject": ref["name"], "variant": variant, "expected": expected,
                   "actual": actual, "status": result["status"], "error": error,
                   "passed": error is None and actual == expected}
            cases.append(row)
            print(json.dumps(row), flush=True)
    report = {"model": config.PHOTO_MEMORY_MODEL or config.VISION_MODEL,
              "scope": "saved-reference sanity only; no held-out camera frames", "cases": cases}
    path = photo_album.root() / "replay_report.json"
    path.write_text(json.dumps(report, indent=2) + "\n")
    raise SystemExit(0 if all(c["passed"] for c in cases) else 1)


if __name__ == "__main__":
    main()
