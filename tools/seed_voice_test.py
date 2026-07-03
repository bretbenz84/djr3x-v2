#!/usr/bin/env python3
"""Audition ElevenLabs v3 SEEDS for Rex's voice — at scale.

The seed picks WHICH vocal character/take v3 settles on (verified: distinct seeds -> distinct
takes). So to change the voice you just try seeds until one sounds right, then set config.TTS_V3_SEED.

Synthesizes the same short line at each seed through the robot's real voice_id / eleven_v3 /
stability, saves one mp3 per seed, and writes an index.html player so you can click through many
fast. Requests are made with an EXPLICIT per-call seed (no global-config mutation), so it's
thread-safe and parallelized.

    ./venv/bin/python -m tools.seed_voice_test                       # default: seeds 1..24
    ./venv/bin/python -m tools.seed_voice_test --range 1 100         # 100 seeds
    ./venv/bin/python -m tools.seed_voice_test --seeds 7 88 2187
    ./venv/bin/python -m tools.seed_voice_test --range 1 100 --line "Say something, Rex."

Output: tts_samples/seeds/seed_<n>.mp3  +  tts_samples/seeds/index.html   (gitignored)
Cost: ElevenLabs v3 bills 1 credit/char, so N seeds x len(line). The default line is short on
purpose. NOTE: previous_text/stitching is unsupported on eleven_v3, so this is one clean generation.
"""

import argparse
import concurrent.futures
import html
import random
import sys
import time
from pathlib import Path

SEED_MAX = 4294967295  # ElevenLabs seed range is 0..4294967295

# Short but characterful — enough to judge the voice, cheap enough to run 100.
DEFAULT_LINE = "Oh good, you're back. Did you miss me?"
# ElevenLabs caps this subscription at 5 concurrent requests — stay UNDER it to avoid 429s.
DEFAULT_WORKERS = 4


def _synth(client, VoiceSettings, voice_id, model_id, vs_dict, text, seed, retries=6):
    """One synthesis with an explicit seed. Retries on the 5-concurrent-request rate limit.
    Returns bytes on success, or a short 'ERROR: ...' string."""
    kwargs = {"voice_id": voice_id, "text": text, "model_id": model_id, "seed": int(seed)}
    if vs_dict:
        kwargs["voice_settings"] = VoiceSettings(**{k: v for k, v in vs_dict.items() if v is not None})
    for attempt in range(retries):
        try:
            return b"".join(client.text_to_speech.stream(**kwargs)) or None
        except Exception as exc:
            msg = str(exc)
            rate_limited = "429" in msg or "concurrent" in msg.lower() or "rate_limit" in msg.lower()
            if rate_limited and attempt < retries - 1:
                time.sleep(1.0 + 1.2 * attempt)   # simple backoff; another worker will free a slot
                continue
            return f"ERROR: {msg[:150]}"


def _write_index(out_dir: Path, line: str, current_seed) -> Path:
    """Rebuild the player from EVERY seed_*.mp3 in the folder (accumulates across runs — this tool
    never deletes prior samples). Newest files first so a fresh batch is at the top."""
    files = list(out_dir.glob("seed_*.mp3"))

    def _seed_of(p: Path):
        try:
            return int(p.stem.split("_", 1)[1])
        except (ValueError, IndexError):
            return -1

    files.sort(key=lambda p: p.stat().st_mtime, reverse=True)  # most recently generated on top
    rows = []
    for p in files:
        seed = _seed_of(p)
        mark = ' <span class="cur">current</span>' if seed == current_seed else ""
        rows.append(
            f'<tr><td>{seed}{mark}</td>'
            f'<td><audio controls preload="none" src="{html.escape(p.name)}"></audio></td>'
            f'<td><button onclick="navigator.clipboard.writeText(\'{seed}\')">copy seed</button></td></tr>'
        )
    doc = f"""<!doctype html><meta charset="utf-8"><title>Rex seed audition</title>
<style>
 body{{font:15px/1.5 system-ui,sans-serif;margin:2rem;max-width:760px}}
 h1{{font-size:1.2rem}} .line{{color:#555;font-style:italic;margin-bottom:1rem}}
 table{{border-collapse:collapse;width:100%}} td{{padding:.4rem .6rem;border-bottom:1px solid #eee;vertical-align:middle}}
 .cur{{background:#ffe08a;border-radius:4px;padding:0 .35rem;font-size:.8rem}}
 .fail{{color:#b00}} audio{{height:32px}}
</style>
<h1>Rex v3 seed audition — {len(rows)} samples (newest first)</h1>
<div class="line">Latest batch line: “{html.escape(line)}”</div>
<table><tr><td><b>seed</b></td><td><b>audio</b></td><td></td></tr>
{chr(10).join(rows)}
</table>
<p style="color:#777;margin-top:1rem">Pick the seed you like and tell Claude the number — it'll set <code>TTS_V3_SEED</code>.</p>
"""
    idx = out_dir / "index.html"
    idx.write_text(doc, encoding="utf-8")
    return idx


def main() -> int:
    ap = argparse.ArgumentParser(description="Audition ElevenLabs v3 seeds for Rex's voice.")
    g = ap.add_mutually_exclusive_group()
    g.add_argument("--seeds", type=int, nargs="+", help="explicit seeds")
    g.add_argument("--range", type=int, nargs=2, metavar=("START", "END"), help="inclusive seed range")
    g.add_argument("--random", type=int, metavar="N",
                   help="N seeds scattered across the FULL 0..4.29B range (varied characters)")
    ap.add_argument("--line", type=str, default=DEFAULT_LINE, help="line to speak (keep it short)")
    ap.add_argument("--workers", type=int, default=DEFAULT_WORKERS, help="concurrent requests")
    args = ap.parse_args()

    import config
    from audio import tts
    import apikeys
    from elevenlabs import ElevenLabs, VoiceSettings

    if str(config.TTS_MODEL_ID).strip() != "eleven_v3":
        print(f"[seed-test] TTS_MODEL_ID is {config.TTS_MODEL_ID!r}, not eleven_v3 — aborting.")
        return 2

    if args.seeds:
        seeds = args.seeds
    elif args.range:
        seeds = list(range(args.range[0], args.range[1] + 1))
    elif args.random:
        seeds = sorted(random.sample(range(0, SEED_MAX + 1), args.random))
        print(f"[seed-test] random seeds: {seeds}")
    else:
        seeds = list(range(1, 25))

    voice_id = config.ELEVENLABS_VOICE_ID
    model_id = config.TTS_MODEL_ID
    vs_dict = tts._resolve_voice_settings("neutral", None)  # pins v3 stability
    current_seed = getattr(config, "TTS_V3_SEED", None)
    client = ElevenLabs(api_key=apikeys.ELEVENLABS_API_KEY)

    out_dir = Path(config.TTS_CACHE_DIR).parent / "tts_samples" / "seeds"
    out_dir.mkdir(parents=True, exist_ok=True)

    # Idempotent gap-fill: skip seeds already generated so a re-run (e.g. after a 429 batch) only
    # pays for the missing ones. Delete the mp3 to force a re-gen.
    results: dict[int, bool] = {}
    todo = []
    for s in seeds:
        p = out_dir / f"seed_{s}.mp3"
        if p.exists() and p.stat().st_size > 0:
            results[s] = True
        else:
            todo.append(s)

    print(f"[seed-test] voice={voice_id} model={model_id} stability={vs_dict.get('stability')}")
    print(f"[seed-test] {len(seeds)} seeds ({len(seeds) - len(todo)} already done) -> generating "
          f"{len(todo)} x {len(args.line)} chars = ~{len(todo) * len(args.line)} credits")
    print(f"[seed-test] line: {args.line!r}  ({args.workers} workers)")

    with concurrent.futures.ThreadPoolExecutor(max_workers=max(1, args.workers)) as pool:
        futs = {
            pool.submit(_synth, client, VoiceSettings, voice_id, model_id, vs_dict, args.line, s): s
            for s in todo
        }
        done = 0
        for fut in concurrent.futures.as_completed(futs):
            seed = futs[fut]
            audio = fut.result()
            done += 1
            if isinstance(audio, bytes):
                (out_dir / f"seed_{seed}.mp3").write_bytes(audio)
                results[seed] = True
            else:
                results[seed] = False
                print(f"[seed-test] seed {seed}: {audio}")
            if done % 10 == 0 or done == len(todo):
                print(f"[seed-test] {done}/{len(todo)} generated")

    ok = sum(1 for s in seeds if results.get(s))
    idx = _write_index(out_dir, args.line, current_seed)   # scans ALL seed_*.mp3 — accumulates
    total = len(list(out_dir.glob("seed_*.mp3")))
    print(f"\n[seed-test] this run: {ok}/{len(seeds)} | folder now holds {total} samples (nothing deleted)")
    print(f"[seed-test] open the player:  {idx}")
    print("[seed-test] pick a seed, tell Claude the number, and it'll set TTS_V3_SEED.")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
