#!/usr/bin/env python3
"""Audition ElevenLabs v3 SEEDS for Rex's voice.

With request stitching in place, the voice is consistent WITHIN a reply regardless of seed —
the seed just picks WHICH vocal character/take v3 settles on. So to change the character you
don't retune anything; you just try seeds until one sounds right, then set config.TTS_V3_SEED.

This synthesizes the same line at each seed through the SAME path the robot uses (current
voice_id, eleven_v3, stability 0.5) and saves one mp3 per seed so you can A/B them, then tell
Claude the winning number.

    ./venv/bin/python -m tools.seed_voice_test                 # default spread + default line
    ./venv/bin/python -m tools.seed_voice_test --seeds 7 88 2187
    ./venv/bin/python -m tools.seed_voice_test --line "Oh good, you're back."

Output: tts_samples/seeds/seed_<n>.mp3   (tts_samples/ is gitignored — local audition only)
Cost: ElevenLabs v3 bills 1 credit/char, so N seeds x len(line) chars. Keep the line short.
"""

import argparse
import sys
from pathlib import Path

# A couple of short sentences with a bit of Rex attitude so the character is audible. The 2nd
# sentence is stitched onto the 1st (previous_text) so you hear the real streamed continuity.
DEFAULT_LINE_1 = "Oh good, you're back."
DEFAULT_LINE_2 = "I was starting to think my charm finally short-circuited something."

# 42 is the current shipped seed — included so you can hear what you DON'T like next to the rest.
DEFAULT_SEEDS = [42, 7, 101, 777, 2187, 55555, 314159]


def main() -> int:
    ap = argparse.ArgumentParser(description="Audition ElevenLabs v3 seeds for Rex's voice.")
    ap.add_argument("--seeds", type=int, nargs="+", default=DEFAULT_SEEDS,
                    help="seeds to try (default: %(default)s)")
    ap.add_argument("--line", type=str, default=None,
                    help="single line to speak (default: a 2-sentence stitched Rex line)")
    ap.add_argument("--line2", type=str, default=None,
                    help="optional 2nd sentence, stitched onto --line (default provided)")
    args = ap.parse_args()

    import config
    from audio import tts

    if str(config.TTS_MODEL_ID).strip() != "eleven_v3":
        print(f"[seed-test] TTS_MODEL_ID is {config.TTS_MODEL_ID!r}, not eleven_v3 — seeds only "
              f"affect v3. Aborting.")
        return 2

    line1 = args.line if args.line is not None else DEFAULT_LINE_1
    line2 = args.line2 if args.line2 is not None else (None if args.line is not None else DEFAULT_LINE_2)

    voice_id = config.ELEVENLABS_VOICE_ID
    model_id = config.TTS_MODEL_ID
    voice_settings = tts._resolve_voice_settings("neutral", None)  # pins v3 stability to 0.5

    out_dir = Path(config.TTS_CACHE_DIR).parent / "tts_samples" / "seeds"
    out_dir.mkdir(parents=True, exist_ok=True)

    total_chars = len(line1) + (len(line2) if line2 else 0)
    print(f"[seed-test] voice={voice_id} model={model_id} stability={voice_settings.get('stability')}")
    print(f"[seed-test] {len(args.seeds)} seeds x ~{total_chars} chars "
          f"= ~{len(args.seeds) * total_chars} credits total")
    print(f"[seed-test] line: {line1!r}" + (f" + {line2!r}" if line2 else ""))

    # NOTE: eleven_v3 rejects previous_text (400 unsupported_model), so we synthesize the whole
    # line in ONE generation — which is also how a consistent reply should be produced on v3.
    full_line = line1 if not line2 else f"{line1} {line2}"

    orig_seed = getattr(config, "TTS_V3_SEED", None)
    ok = 0
    try:
        for seed in args.seeds:
            config.TTS_V3_SEED = seed  # _fetch_from_api reads this
            audio = tts._fetch_from_api(full_line, voice_id, model_id, voice_settings)
            if not audio:
                print(f"[seed-test] seed {seed}: FAILED (no audio — check credits / API)")
                continue
            path = out_dir / f"seed_{seed}.mp3"
            path.write_bytes(audio)
            marker = "  <- current" if seed == orig_seed else ""
            print(f"[seed-test] seed {seed:>7}  ->  {path}{marker}")
            ok += 1
    finally:
        config.TTS_V3_SEED = orig_seed

    print(f"\n[seed-test] wrote {ok}/{len(args.seeds)} samples to {out_dir}")
    print("[seed-test] listen, then tell Claude the seed you like and it'll set TTS_V3_SEED.")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
