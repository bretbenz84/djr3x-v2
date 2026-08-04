#!/usr/bin/env python3
"""Build assets/voices/famous/<slug>.{wav,txt} voice references for deceased US
presidents, from public-domain archive audio.

The clips themselves are gitignored (assets/voices/* except rex/), so this
script is the tracked artifact: run it and the reference set rebuilds from the
original archive sources. See docs/presidential_voice_refs.md for provenance
and the licensing basis.

    venv/bin/python tools/build_presidential_refs.py            # download + build
    venv/bin/python tools/build_presidential_refs.py --no-fetch # rebuild from cache

Each SPAN was hand-picked from a word-timestamped Whisper scan of the source so
that it is CONTIGUOUS SOLO SPEECH BY THE PRESIDENT HIMSELF. Several sources open
with someone else — a radio announcer (Eisenhower), the Chief Justice
administering the oath (Kennedy, Ford), or a modern archive narrator tacked onto
the end (Theodore Roosevelt) — and cloning those would put the wrong voice
behind the president's name. Do not widen a span without re-scanning it.

Pipeline: trim -> loudnorm -> 24 kHz mono PCM_16 -> pad tail -> re-transcribe the
FINAL file, so the .txt provably describes the audio the cloner is given (which
is what a reference transcript is for).
"""
from __future__ import annotations

import argparse
import os
import subprocess
import sys
import urllib.request

import numpy as np
import soundfile as sf

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT = os.path.join(ROOT, "assets", "voices", "famous")
CACHE = os.path.join(ROOT, "assets", "voices", "_src_presidents")
WHISPER = os.path.join(ROOT, "assets", "models", "whisper")
SR = 24000
TAIL_PAD = 0.5
UA = "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 Chrome/120"
MC = "https://d4q9blt8qjhv3.cloudfront.net/audio"
LOC = "https://tile.loc.gov/storage-services/master/mbrsrs/mbrsjukebox"
JUKEBOX = "https://tile.loc.gov/streaming-services/iiif/service:mbrsrs:mbrsjukebox"

# slug, cache filename, source URL, start, end, description
SPANS = [
    ("theodore-roosevelt", "roosevelt-teddy.flac",
     "https://archive.org/download/roosvelt1901/roosvelt1901.flac",
     9.60, 31.86, "1901 speech excerpt (MSU Vincent Voice Library)"),
    ("william-taft", "taft.wav",
     f"{LOC}/ucsb_victor_35255_01_c12444_01/ucsb_victor_35255_01_c12444_01.wav",
     1.26, 18.94, "1912 Victor disc, 'President Taft on Prosperity'"),
    ("woodrow-wilson", "wilson.mp3",
     f"{JUKEBOX}:ucsb_victor_35251_01_c12389_01:ucsb_victor_35251_01_c12389_01"
     "/full/full/0/full/default.mp3",
     20.92, 42.68, "1912 Victor disc, 'Woodrow Wilson on the Trusts'"),
    ("herbert-hoover", "hoover.flac",
     "https://archive.org/download/herbhoov1932/herbhoov1932.flac",
     0.82, 23.12, "1932 campaign address"),
    ("franklin-roosevelt", "roosevelt-fdr.mp3", f"{MC}/spe_1933_0312_roosevelt.mp3",
     39.22, 55.58, "1933 Fireside Chat 1, on the banking crisis"),
    ("harry-truman", "truman.mp3", f"{MC}/spe_1951_0411_truman.mp3",
     3.58, 23.10, "1951 Report to the American People on Korea"),
    ("dwight-eisenhower", "eisenhower.mp3", f"{MC}/spe_1961_0117_eisenhower.mp3",
     13.14, 33.44, "1961 Farewell Address (skips the announcer at 4-12 s)"),
    ("john-kennedy", "kennedy.mp3", f"{MC}/spe_1961_0120_kennedy.mp3",
     80.52, 97.94, "1961 Inaugural (skips the oath, administered 0-31 s)"),
    ("lyndon-johnson", "johnson.mp3", f"{MC}/spe_1965_0315_johnson.mp3",
     299.70, 319.54, "1965 'We Shall Overcome' voting-rights address"),
    ("richard-nixon", "nixon.mp3", f"{MC}/spe_1974_0808_nixon.mp3",
     2.74, 23.30, "1974 Resignation Address"),
    ("gerald-ford", "ford.mp3", f"{MC}/spe_1974_0809_ford.mp3",
     146.50, 163.40, "1974 oath remarks (skips the oath, administered 14-59 s)"),
    ("jimmy-carter", "carter.mp3", f"{MC}/spe_1979_0715_carter.mp3",
     8.68, 28.92, "1979 'Crisis of Confidence' address"),
    ("ronald-reagan", "reagan.mp3", f"{MC}/spe_1987_0612_reagan.mp3",
     177.80, 199.72, "1987 Brandenburg Gate address"),
    ("george-hw-bush", "bush41.mp3", f"{MC}/spe_1989_0120_bush.mp3",
     933.36, 954.74, "1989 Inaugural Address"),
]

# Nickname -> canonical slug. find_famous_ref() matches an exact slug first and
# then loosely on the SURNAME token, so "FDR"/"JFK"/"Ike" (no surname in them)
# would otherwise miss entirely. Symlinks, not copies.
ALIASES = {
    "fdr": "franklin-roosevelt",
    "teddy-roosevelt": "theodore-roosevelt",
    "jfk": "john-kennedy",
    "lbj": "lyndon-johnson",
    "ike": "dwight-eisenhower",
}


def fetch(url: str, dst: str) -> None:
    if os.path.exists(dst) and os.path.getsize(dst) > 10_000:
        return
    req = urllib.request.Request(url, headers={
        "User-Agent": UA,
        # Miller Center's CloudFront 403s unless a referer is present, and it
        # rate-limits bursts — a rebuild that trips it should wait, not retry hard.
        "Referer": "https://millercenter.org/",
    })
    with urllib.request.urlopen(req, timeout=120) as r, open(dst, "wb") as fh:
        fh.write(r.read())


def build(slug: str, src: str, start: float, end: float, transcribe) -> tuple[float, str]:
    dst = os.path.join(OUT, f"{slug}.wav")
    subprocess.run(
        ["ffmpeg", "-y", "-v", "error",
         "-ss", str(start), "-to", str(end), "-i", src,
         "-ac", "1", "-ar", str(SR),
         "-af", "loudnorm=I=-18:TP=-2:LRA=11,afade=t=in:st=0:d=0.03",
         "-c:a", "pcm_s16le", dst],
        check=True,
    )
    arr, sr = sf.read(dst, dtype="float32")
    arr = np.concatenate([arr, np.zeros(int(TAIL_PAD * sr), dtype=np.float32)])
    sf.write(dst, arr, sr, subtype="PCM_16")

    text = " ".join(transcribe(dst).split())
    with open(os.path.join(OUT, f"{slug}.txt"), "w", encoding="utf-8") as fh:
        fh.write(text)
    return len(arr) / sr, text


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--no-fetch", action="store_true",
                    help="rebuild from already-downloaded sources only")
    args = ap.parse_args()

    os.makedirs(OUT, exist_ok=True)
    os.makedirs(CACHE, exist_ok=True)

    import mlx_whisper

    def transcribe(path: str) -> str:
        return mlx_whisper.transcribe(
            path, path_or_hf_repo=WHISPER, condition_on_previous_text=False
        )["text"]

    built = 0
    for slug, fname, url, start, end, note in SPANS:
        src = os.path.join(CACHE, fname)
        if not args.no_fetch:
            try:
                fetch(url, src)
            except Exception as exc:
                print(f"  {slug}: fetch failed ({exc})", file=sys.stderr)
        if not os.path.exists(src):
            print(f"  {slug}: no source, skipped", file=sys.stderr)
            continue
        dur, text = build(slug, src, start, end, transcribe)
        built += 1
        print(f"{slug:20} {dur:5.2f}s  {note}")
        print(f"{'':20} {text[:96]}")

    for alias, target in ALIASES.items():
        for ext in ("wav", "txt"):
            link = os.path.join(OUT, f"{alias}.{ext}")
            if os.path.lexists(link):
                os.unlink(link)
            if os.path.exists(os.path.join(OUT, f"{target}.{ext}")):
                os.symlink(f"{target}.{ext}", link)

    print(f"\n{built}/{len(SPANS)} references in {OUT}")
    return 0 if built == len(SPANS) else 1


if __name__ == "__main__":
    raise SystemExit(main())
