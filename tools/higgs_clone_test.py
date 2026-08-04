#!/usr/bin/env python3
"""
tools/higgs_clone_test.py — Higgs Audio v3 (4B) voice-clone quality bench.

Records you reading a passage off the terminal, clones your voice with the
on-device Higgs Audio v3 TTS model, and plays back a short R3X-style self-roast
in your voice — reporting exactly how long every stage took.

Why this exists: the shipped impersonation engine (Qwen3-TTS 1.7B-8bit, see
audio/local_tts.py) clones from an ~8 s far-field 16 kHz mic capture and the
result doesn't sound like the target. Two variables are confounded there — the
model AND the reference clip. This bench separates them: a bigger model, and a
reference you control the length and quality of.

    python tools/higgs_clone_test.py                     # record, clone, roast
    python tools/higgs_clone_test.py --ref-secs 0        # feed the WHOLE take
    python tools/higgs_clone_test.py --ref-secs 15       # short reference
    python tools/higgs_clone_test.py --rate 16000        # mimic the robot's mic
    python tools/higgs_clone_test.py --ref assets/voices/people/1.wav
                                                         # reuse an existing clip
    python tools/higgs_clone_test.py --repeat 3          # 3 takes, timing spread
    python tools/higgs_clone_test.py --list-devices

REFERENCE LENGTH IS THE VARIABLE TO PLAY WITH. Longer is not automatically
better: the reference is encoded at 25 frames/sec into an 8192-token context, so
a 90 s clip both crowds the context and slows every generation. Default is a
30 s window taken from the middle of your recording (the steadiest part — you've
settled in, and you haven't hit the trailing "am I done?" mumble).

Standalone: needs config + a mic + the model. No Rex stack, no LEDs, no servos.
"""

import argparse
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

import numpy as np
import sounddevice as sd
import soundfile as sf

import config

# Where setup drops the weights. bf16 is ~9.3 GB resident — fine on a 24 GB dev
# Mac, NOT fine on the robot's 16 GB M2 alongside whisper/qwen-asr/vision. An
# 8-bit conversion is the robot-side answer; this bench measures the ceiling.
DEFAULT_MODEL_DIR = "assets/models/higgs_tts/4b-bf16"
MODEL_REPO = "bosonai/higgs-audio-v3-tts-4b"

# ── The passage ──────────────────────────────────────────────────────────────
# ~240 words ≈ 90 s at a normal reading pace. Deliberately varied: statements,
# questions, an exclamation, numbers, and a spread of vowels and plosives — a
# clone conditioned on flat monotone reference audio produces flat monotone
# speech, so the passage has to make you move.

PASSAGE = """\
Okay, let's get this over with. My name is on the build log, the soldering iron
is still warm, and somewhere under this workbench there is a screw I will never
find again.

Here is the thing about droids. They do not care how late it is. They do not
care that you said "one more test" four hours ago. You point a camera at the
world, you give the thing a voice, and suddenly it has opinions about you.

How did I get here? Honestly? I wanted a robot that could say hello. That was
the whole plan. Hello. Two syllables. Now there are twelve thousand lines of
code, six microphones, a neck that moves in three directions, and a face
recognition model that keeps insisting the houseplant is my cousin.

Was it worth it? Ask me tomorrow, when the servos stop buzzing.

But listen — when it works, it really works. You walk into the room, and it
turns to look at you. Not at the wall. Not at the lamp. At you. It knows your
name, it remembers what you told it last Tuesday, and it makes a joke about it
before you've finished taking your coat off.

That is the part nobody warns you about. You build a machine to be useful, and
somewhere along the way it becomes company.

Anyway. That's enough talking. Let's hear how badly this thing mangles my
voice."""

# ── The roast ────────────────────────────────────────────────────────────────
# First person, as the speaker — same shape as features/impersonation.py's
# generated scripts, so the bench exercises the real use case. Venue-neutral
# (Rex is usually not in a cantina) and affectionate, punching at the quirk.

ROAST_LINES = [
    "Yeah, I built a droid that roasts me. No, I have not thought about why. "
    "Next question.",

    "I've rewired this neck servo six times tonight. Six! And it still looks at "
    "me like I'm the problem.",

    "Sure, I could go outside. Or I could spend another weekend teaching a robot "
    "to do my voice back at me.",

    "I said I'd stop after one more test. That was four hours ago, and I am "
    "still holding a soldering iron.",
]


# ── Audio device plumbing (mirrors tools/test_voice_id.py) ───────────────────

def _input_candidates() -> "list[tuple[object, int]]":
    """(device, channels) pairs to try, in order. macOS device indices float
    between boots and query_devices over-reports channels, so the only reliable
    test is trying to open the stream — see audio/stream.py."""
    configured = getattr(config, "AUDIO_DEVICE_INDEX", None)
    requested = int(
        getattr(config, "AUDIO_INPUT_CHANNELS", getattr(config, "AUDIO_CHANNELS", 1)) or 1
    )
    out = []
    for device in (configured, None):
        for ch in (requested, 2, 1):
            if (device, ch) not in out:
                out.append((device, ch))
    return out


def record_interactive(rate: int, max_secs: float) -> np.ndarray:
    """Record until the user presses Enter (or max_secs). Returns float32 mono.

    Uses a callback stream rather than sd.rec() so the take is open-ended — you
    read at your own pace instead of racing a fixed countdown, which is exactly
    the thing that produced a rushed 8-second reference last time.
    """
    frames: "list[np.ndarray]" = []
    stop = threading.Event()

    def _cb(indata, _n, _t, status):
        if status:
            pass  # overflows are survivable here; the meter will show the dropout
        frames.append(indata.copy())

    last_exc = None
    for device, ch in _input_candidates():
        try:
            stream = sd.InputStream(
                samplerate=rate, channels=ch, dtype="float32",
                device=device, callback=_cb, blocksize=1024,
            )
            stream.start()
        except Exception as exc:
            last_exc = exc
            continue

        threading.Thread(
            target=lambda: (input(), stop.set()), daemon=True
        ).start()

        t0 = time.monotonic()
        try:
            while not stop.is_set():
                elapsed = time.monotonic() - t0
                if elapsed >= max_secs:
                    print("\n  (hit --secs limit)")
                    break
                level = 0.0
                if frames:
                    tail = frames[-1]
                    level = float(np.sqrt(np.mean(np.square(tail))))
                bars = int(min(level * 240, 30))
                sys.stdout.write(
                    f"\r  recording {elapsed:5.1f}s  "
                    f"[{'#' * bars}{' ' * (30 - bars)}]  press Enter to stop "
                )
                sys.stdout.flush()
                time.sleep(0.08)
        finally:
            stream.stop()
            stream.close()
        print()

        if not frames:
            sys.exit("no audio captured")
        audio = np.concatenate(frames, axis=0)
        if audio.ndim > 1:                     # downmix whatever we opened
            audio = audio.mean(axis=1)
        return np.ascontiguousarray(audio.astype(np.float32).reshape(-1))

    sys.exit(f"could not open any input device: {last_exc}")


def describe(audio: np.ndarray, rate: int, label: str) -> None:
    peak = float(np.max(np.abs(audio))) if audio.size else 0.0
    rms = float(np.sqrt(np.mean(np.square(audio)))) if audio.size else 0.0
    dbfs = 20 * np.log10(max(peak, 1e-9))
    rdbfs = 20 * np.log10(max(rms, 1e-9))
    print(f"  {label}: {audio.size / rate:.1f}s @ {rate} Hz  "
          f"peak {dbfs:+.1f} dBFS  rms {rdbfs:+.1f} dBFS")
    if dbfs < -20:
        print("    ⚠ very quiet — move closer to the mic or raise AUDIO_INPUT_GAIN")
    elif dbfs > -0.5:
        print("    ⚠ clipping — back off the mic; a clipped reference clones badly")


def center_window(audio: np.ndarray, rate: int, secs: float) -> np.ndarray:
    """Take `secs` from the middle of the take. The middle is the best part: past
    the throat-clear, before the trailing mumble."""
    want = int(secs * rate)
    if want <= 0 or audio.size <= want:
        return audio
    start = (audio.size - want) // 2
    return audio[start:start + want]


# ── Model ────────────────────────────────────────────────────────────────────

def ensure_model(model_dir: Path) -> Path:
    if (model_dir / "model.safetensors").exists():
        return model_dir
    print(f"  weights not found at {model_dir}")
    print(f"  fetching {MODEL_REPO} (~9.3 GB) — one time, then cached ...")
    from huggingface_hub import snapshot_download
    snapshot_download(repo_id=MODEL_REPO, local_dir=str(model_dir), max_workers=8)
    return model_dir


def load_higgs(model_dir: Path):
    """Load offline from the local dir. Scoped HF offline flags for the same
    reason audio/local_tts.py uses them: post_load_hook builds a transformers
    tokenizer whose only network reach we want closed, without leaking the flags
    to the rest of the process."""
    saved = {k: os.environ.get(k) for k in ("HF_HUB_OFFLINE", "TRANSFORMERS_OFFLINE")}
    os.environ["HF_HUB_OFFLINE"] = "1"
    os.environ["TRANSFORMERS_OFFLINE"] = "1"
    try:
        from mlx_audio.tts.utils import load_model
        # model_type is passed EXPLICITLY on purpose. mlx-audio 0.4.5 resolves
        # the architecture in two stages: it looks config.json's
        # "higgs_multimodal_qwen3" up in MODEL_REMAPPING (correctly getting
        # "higgs_audio_v3"), and then throws that result away in favour of
        # scanning the model PATH for a recognizable segment. Loading from a
        # local dir named "4b-bf16" leaves nothing to match, so it tries to
        # import mlx_audio.tts.models.higgs_multimodal_qwen3 and dies. Naming
        # the type here skips the path-sniffing entirely.
        return load_model(str(model_dir), model_type="higgs_audio_v3")
    finally:
        for k, v in saved.items():
            if v is None:
                os.environ.pop(k, None)
            else:
                os.environ[k] = v


def transcribe_ref(audio: np.ndarray, rate: int) -> "str | None":
    """Whisper the reference so ref_text describes what was ACTUALLY said. Worth
    it when the reader paraphrases or stumbles — a ref_text that disagrees with
    the audio degrades the clone."""
    try:
        from scipy.signal import resample_poly
        from audio.transcription import transcribe
        at16k = (audio if rate == 16000
                 else resample_poly(audio, 16000, rate).astype(np.float32))
        text = str(transcribe(np.ascontiguousarray(at16k))).strip()
        return text or None
    except Exception as exc:
        print(f"  (transcription unavailable: {exc} — using the passage text)")
        return None


# ── Main ─────────────────────────────────────────────────────────────────────

def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--secs", type=float, default=180.0,
                    help="hard cap on recording length (default 180)")
    ap.add_argument("--rate", type=int, default=48000,
                    help="capture sample rate (default 48000; use 16000 to mimic the robot)")
    ap.add_argument("--ref-secs", type=float, default=30.0,
                    help="seconds of reference fed to the model, from the middle "
                         "of the take. 0 = use everything (default 30)")
    ap.add_argument("--ref", type=str, default=None,
                    help="skip recording; clone from this WAV instead")
    ap.add_argument("--ref-text", type=str, default=None,
                    help="transcript of --ref (defaults to a sibling .txt, then Whisper)")
    ap.add_argument("--text", type=str, default=None,
                    help="override the roast line")
    ap.add_argument("--repeat", type=int, default=1,
                    help="generate N takes to see the timing spread (default 1)")
    ap.add_argument("--transcribe", action="store_true",
                    help="Whisper the recording for ref_text instead of using the passage")
    ap.add_argument("--temperature", type=float, default=1.0)
    ap.add_argument("--model-dir", type=str, default=DEFAULT_MODEL_DIR)
    ap.add_argument("--no-play", action="store_true", help="write the WAV, don't play it")
    ap.add_argument("--list-devices", action="store_true")
    args = ap.parse_args()

    if args.list_devices:
        print(sd.query_devices())
        return

    out_dir = _ROOT / "assets" / "voices" / "bench"
    out_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 72)
    print("Higgs Audio v3 (4B) — voice clone bench")
    print("=" * 72)

    # ── 1. Reference audio ───────────────────────────────────────────────────
    if args.ref:
        ref_path = Path(args.ref)
        if not ref_path.exists():
            sys.exit(f"no such file: {ref_path}")
        audio, rate = sf.read(str(ref_path))
        audio = np.asarray(audio, dtype=np.float32)
        if audio.ndim > 1:
            audio = audio.mean(axis=1)
        audio = np.ascontiguousarray(audio.reshape(-1))
        print(f"\nReference: {ref_path}")
        describe(audio, rate, "loaded")
        ref_text = args.ref_text
        if ref_text is None:
            sidecar = ref_path.with_suffix(".txt")
            if sidecar.exists():
                ref_text = " ".join(sidecar.read_text(encoding="utf-8").split())
                print(f"  ref_text from {sidecar.name}")
        if ref_text is None:
            ref_text = transcribe_ref(audio, rate)
        if not ref_text:
            sys.exit("need a transcript for the reference (--ref-text)")
    else:
        print("\nRead this out loud, at a normal pace. Punch the punctuation —")
        print("a flat monotone reference produces a flat monotone clone.\n")
        print("-" * 72)
        print(PASSAGE)
        print("-" * 72)
        print("\nAim for at least 60 seconds. Press Enter when you're ready to start,")
        print("then Enter again when you've finished reading.")
        try:
            input()
        except (EOFError, KeyboardInterrupt):
            return
        print()
        audio = record_interactive(args.rate, args.secs)
        rate = args.rate
        describe(audio, rate, "captured")
        if audio.size / rate < 20:
            print("    ⚠ that's a short take — clone quality will suffer")

        ref_text = None
        if args.transcribe:
            print("  transcribing for an exact ref_text ...")
            ref_text = transcribe_ref(audio, rate)
        if not ref_text:
            ref_text = " ".join(PASSAGE.split())

        raw_path = out_dir / "bench_reference_full.wav"
        sf.write(str(raw_path), audio, rate, subtype="PCM_16")
        print(f"  saved full take → {raw_path}")

    # Trim to the reference window actually handed to the model.
    if args.ref_secs and audio.size / rate > args.ref_secs:
        audio = center_window(audio, rate, args.ref_secs)
        # ref_text now over-describes the audio; Whisper the window so the
        # transcript matches what the model actually hears.
        print(f"  trimming to a {args.ref_secs:.0f}s window from the middle")
        trimmed_text = transcribe_ref(audio, rate)
        if trimmed_text:
            ref_text = trimmed_text
        else:
            print("    ⚠ could not re-transcribe the window — ref_text may over-describe "
                  "the audio. Use --ref-secs 0 or --transcribe for a clean pairing.")

    ref_path = out_dir / "bench_reference.wav"
    sf.write(str(ref_path), audio, rate, subtype="PCM_16")
    print(f"\nReference handed to the model: {audio.size / rate:.1f}s → {ref_path}")
    print(f"  ref_text ({len(ref_text.split())} words): {ref_text[:110]}"
          f"{'...' if len(ref_text) > 110 else ''}")

    # ── 2. Model ─────────────────────────────────────────────────────────────
    model_dir = ensure_model(Path(args.model_dir))
    print(f"\nLoading Higgs Audio v3 from {model_dir} ...")
    t0 = time.monotonic()
    model = load_higgs(model_dir)
    t_load = time.monotonic() - t0
    print(f"  model loaded in {t_load:.1f}s")

    # Encode the reference once — Higgs exposes this precisely so repeated
    # generations for the same person skip re-encoding. That's the shape the
    # real feature would use (encode at capture time, cache the codes).
    t0 = time.monotonic()
    try:
        ref_codes = model.encode_reference_audio(str(ref_path))
        t_encode = time.monotonic() - t0
        print(f"  reference encoded in {t_encode:.1f}s")
    except Exception as exc:
        print(f"  (encode_reference_audio unavailable: {exc}; passing the wav directly)")
        ref_codes, t_encode = None, 0.0

    # ── 3. Generate ──────────────────────────────────────────────────────────
    import random
    timings = []
    for i in range(max(1, args.repeat)):
        line = args.text or random.choice(ROAST_LINES)
        print(f"\n── take {i + 1}/{args.repeat} " + "─" * 46)
        print(f'  "{line}"')

        kwargs = dict(text=line, ref_text=ref_text,
                      temperature=args.temperature, max_new_tokens=2048)
        if ref_codes is not None:
            kwargs["ref_audio_codes"] = ref_codes
        else:
            kwargs["ref_audio"] = str(ref_path)

        t0 = time.monotonic()
        result = next(model.generate(**kwargs))
        gen = time.monotonic() - t0

        out = np.asarray(result.audio, dtype=np.float32).reshape(-1)
        sr = int(getattr(result, "sample_rate", 24000))
        dur = out.size / sr if sr else 0.0
        rtf = gen / dur if dur else float("nan")
        timings.append((gen, dur, rtf))
        print(f"  generated {dur:.1f}s of audio in {gen:.1f}s  (RTF {rtf:.2f}"
              f"{'  — slower than realtime' if rtf > 1 else ''})")

        wav = out_dir / f"bench_take{i + 1}.wav"
        sf.write(str(wav), out, sr, subtype="PCM_16")
        print(f"  → {wav}")

        if not args.no_play:
            sd.play(out, sr, blocksize=int(getattr(config, "AUDIO_PLAYBACK_BLOCKSIZE", 4096)),
                    latency=str(getattr(config, "AUDIO_PLAYBACK_LATENCY", "high") or "high"))
            sd.wait()

    # ── 4. Summary ───────────────────────────────────────────────────────────
    print("\n" + "=" * 72)
    print("TIMING")
    print("=" * 72)
    print(f"  model load          {t_load:6.1f}s   (one time per process)")
    print(f"  reference encode    {t_encode:6.1f}s   (one time per person — cacheable)")
    gens = [g for g, _, _ in timings]
    rtfs = [r for _, _, r in timings]
    print(f"  generation          {min(gens):6.1f}s"
          f"{f' – {max(gens):.1f}s' if len(gens) > 1 else '        '}"
          f"   RTF {min(rtfs):.2f}"
          f"{f' – {max(rtfs):.2f}' if len(rtfs) > 1 else ''}")
    print()
    print(f"  For comparison, the shipped Qwen3-TTS 1.7B-8bit engine generates a")
    print(f"  ~13 s take in ~16 s cold (RTF ~1.0–1.2) at 2.4 GB resident.")
    print(f"  This bench is bf16 (~9.3 GB) — the robot's 16 GB M2 needs an 8-bit")
    print(f"  conversion (~4.7 GB) before this ships.")
    print()
    print(f"  Reference used: {audio.size / rate:.1f}s @ {rate} Hz")
    print(f"  Re-run with --ref-secs 60 / --ref-secs 15 to find where quality plateaus,")
    print(f"  and --rate 16000 to see what the robot's mic actually costs you.")
    print()


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\ninterrupted")
