#!/usr/bin/env python3
"""
tools/asr_bench.py — offline ASR shoot-out on THIS ROOM's actual audio.

Replays the WAV takes saved by tools/mic_check.py (score mode) through two
engines side by side and scores word accuracy against the reference sentences
recorded in logs/mic_check/history.jsonl:

    A. the robot's current backend  — mlx-whisper via audio/transcription.py
       (the full live path: same model, same decoder bias)
    B. a candidate model            — Qwen3-ASR via mlx_audio (already in the
       venv for Rex's local TTS; weights fetched on first run, ~2 GB for 8bit)

No microphone, no robot, no recording — the comparison runs on identical bytes,
so the verdict is about the MODELS, not the room. Run it after every score
session to grow the evidence base.

    ./venv/bin/python tools/asr_bench.py                 # all takes in history
    ./venv/bin/python tools/asr_bench.py --limit 10      # newest 10 takes only
    ./venv/bin/python tools/asr_bench.py --model mlx-community/Qwen3-ASR-0.6B-8bit

Weights land in assets/models/qwen_asr/ (gitignored, like every other model).
"""

from __future__ import annotations

import argparse
import json
import sys
import time
import wave
from pathlib import Path

import numpy as np

_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_ROOT))

DEFAULT_MODEL = "mlx-community/Qwen3-ASR-1.7B-8bit"
MODEL_DIR = _ROOT / "assets" / "models" / "qwen_asr"
HISTORY = _ROOT / "logs" / "mic_check" / "history.jsonl"


def _load_wav(path: Path) -> np.ndarray:
    with wave.open(str(path), "rb") as w:
        assert w.getnchannels() == 1 and w.getsampwidth() == 2, path
        pcm = np.frombuffer(w.readframes(w.getnframes()), dtype=np.int16)
    return (pcm.astype(np.float32) / 32768.0)


def _word_accuracy(ref: str, hyp: str) -> float:
    # Same scoring as tools/mic_check.py (kept in sync by tests/test_asr_bench.py).
    import re

    tok = lambda s: re.sub(r"[^a-z0-9' ]", " ", s.lower()).split()
    r, h = tok(ref), tok(hyp)
    if not r:
        return 0.0
    d = np.zeros((len(r) + 1, len(h) + 1), dtype=np.int32)
    d[:, 0] = np.arange(len(r) + 1)
    d[0, :] = np.arange(len(h) + 1)
    for i in range(1, len(r) + 1):
        for j in range(1, len(h) + 1):
            d[i, j] = min(d[i - 1, j] + 1, d[i, j - 1] + 1,
                          d[i - 1, j - 1] + (r[i - 1] != h[j - 1]))
    return max(0.0, 1.0 - float(d[len(r), len(h)]) / len(r))


def _takes(limit: int | None) -> list[tuple[Path, str]]:
    """(wav_path, reference) pairs from every score run, newest last."""
    if not HISTORY.exists():
        raise SystemExit(f"{HISTORY} not found — run `tools/mic_check.py score` first.")
    pairs: list[tuple[Path, str]] = []
    for line in HISTORY.read_text().splitlines():
        try:
            rec = json.loads(line)
        except json.JSONDecodeError:
            continue
        for i, row in enumerate(rec.get("rows") or [], 1):
            wav = HISTORY.parent / f"score-{rec['ts']}-{i}.wav"
            if wav.exists() and row.get("ref"):
                pairs.append((wav, str(row["ref"])))
    if not pairs:
        raise SystemExit("history.jsonl has no rows with surviving WAV files.")
    return pairs[-limit:] if limit else pairs


def _whisper_backend():
    from audio import transcription

    def run(mono: np.ndarray) -> str:
        return str(transcription.transcribe(mono) or "").strip()
    return run


def _qwen_backend(repo: str):
    from huggingface_hub import snapshot_download
    from mlx_audio.stt.utils import load_model

    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    local = snapshot_download(repo, local_dir=MODEL_DIR / repo.split("/")[-1])
    model = load_model(local)

    def run(mono: np.ndarray) -> str:
        out = model.generate(mono, language="en", verbose=False)
        return str(getattr(out, "text", out) or "").strip()
    return run


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", default=DEFAULT_MODEL,
                    help=f"candidate HF repo (default {DEFAULT_MODEL})")
    ap.add_argument("--limit", type=int, default=None, help="newest N takes only")
    args = ap.parse_args()

    takes = _takes(args.limit)
    print(f"{len(takes)} takes from {HISTORY}")
    print(f"candidate: {args.model}\n")

    print("loading whisper backend (robot's live path)...")
    whisper = _whisper_backend()
    print("loading candidate (downloads weights on first run)...")
    qwen = _qwen_backend(args.model)

    rows = []
    for wav, ref in takes:
        mono = _load_wav(wav)
        t0 = time.perf_counter()
        w_hyp = whisper(mono)
        w_secs = time.perf_counter() - t0
        t0 = time.perf_counter()
        q_hyp = qwen(mono)
        q_secs = time.perf_counter() - t0
        rows.append({
            "wav": wav.name, "ref": ref,
            "whisper": {"hyp": w_hyp, "acc": _word_accuracy(ref, w_hyp), "secs": w_secs},
            "qwen": {"hyp": q_hyp, "acc": _word_accuracy(ref, q_hyp), "secs": q_secs},
        })
        print(f"  {wav.name}")
        print(f"    ref     : {ref!r}")
        print(f"    whisper : {w_hyp!r}  ({rows[-1]['whisper']['acc'] * 100:.0f}%, {w_secs:.2f}s)")
        print(f"    qwen    : {q_hyp!r}  ({rows[-1]['qwen']['acc'] * 100:.0f}%, {q_secs:.2f}s)")

    wa = float(np.mean([r["whisper"]["acc"] for r in rows]))
    qa = float(np.mean([r["qwen"]["acc"] for r in rows]))
    wt = float(np.median([r["whisper"]["secs"] for r in rows]))
    qt = float(np.median([r["qwen"]["secs"] for r in rows]))
    print("\n" + "=" * 68)
    print(f"  {'':>10}  {'mean accuracy':>14}  {'median latency':>15}")
    print(f"  {'whisper':>10}  {wa * 100:>13.1f}%  {wt:>14.2f}s")
    print(f"  {'qwen3':>10}  {qa * 100:>13.1f}%  {qt:>14.2f}s")
    print("=" * 68)
    if qa > wa + 0.02:
        print("  Qwen3-ASR is meaningfully MORE accurate on this room's audio.")
    elif wa > qa + 0.02:
        print("  Whisper is meaningfully more accurate — keep it.")
    else:
        print("  Accuracy is a wash — decide on latency and robustness instead.")

    out = HISTORY.parent / "asr_bench_results.jsonl"
    with out.open("a") as f:
        f.write(json.dumps({"ts": time.strftime("%Y%m%d-%H%M%S"),
                            "model": args.model, "whisper_acc": round(wa, 3),
                            "qwen_acc": round(qa, 3), "whisper_median_secs": round(wt, 2),
                            "qwen_median_secs": round(qt, 2), "n": len(rows),
                            "rows": rows}) + "\n")
    print(f"  appended to {out}")


if __name__ == "__main__":
    main()
