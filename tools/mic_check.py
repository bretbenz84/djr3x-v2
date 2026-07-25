#!/usr/bin/env python3
"""
tools/mic_check.py — Far-field microphone diagnostic for the ReSpeaker Lite.

Answers the question "why are transcriptions bad when I talk from across the
room?" with measurements instead of guesses. It reproduces the LIVE capture math
(audio/stream.py: channel selection, then AUDIO_INPUT_GAIN with hard clipping),
so what it reports is what Whisper actually receives.

    ./venv/bin/python tools/mic_check.py channels     # what is on each mic channel
    ./venv/bin/python tools/mic_check.py noise        # room noise floor (stay quiet)
    ./venv/bin/python tools/mic_check.py speech       # speak from your normal spot
    ./venv/bin/python tools/mic_check.py distance     # guided 3ft / 6ft / 9ft sweep
    ./venv/bin/python tools/mic_check.py transcribe   # end-to-end: hear it as Rex does
    ./venv/bin/python tools/mic_check.py all          # channels + noise + speech + verdict

Nothing here writes to the robot's databases or speaks. Run it with the main app
STOPPED (it needs exclusive use of the mic).

Reference points used for the verdicts (16-bit capture, speech at conversational
level): a healthy far-field signal lands around -30 dBFS RMS with >= 18 dB SNR.
Whisper degrades quickly below ~12 dB SNR, and clipping above ~1% of samples
hurts more than low level does.
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np

_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_ROOT))

import config  # noqa: E402
from utils.config_loader import AUDIO_DEVICE_INDEX  # noqa: E402

SR = int(getattr(config, "AUDIO_SAMPLE_RATE", 16000))

# Verdict thresholds — see module docstring.
GOOD_SNR_DB = 18.0
POOR_SNR_DB = 12.0
TARGET_SPEECH_DBFS = -30.0
CLIP_WARN_FRAC = 0.01


# ── helpers ───────────────────────────────────────────────────────────────────

def _dbfs(x: np.ndarray) -> float:
    """RMS in dBFS. -inf-safe: digital silence reports -120."""
    if x.size == 0:
        return -120.0
    rms = float(np.sqrt(np.mean(np.square(x.astype(np.float64)))))
    return 20.0 * np.log10(rms) if rms > 1e-6 else -120.0


def _peak_dbfs(x: np.ndarray) -> float:
    if x.size == 0:
        return -120.0
    peak = float(np.max(np.abs(x)))
    return 20.0 * np.log10(peak) if peak > 1e-6 else -120.0


def _clip_fraction(x: np.ndarray) -> float:
    if x.size == 0:
        return 0.0
    return float(np.mean(np.abs(x) >= 0.999))


def _device_index() -> int | None:
    if AUDIO_DEVICE_INDEX is not None:
        return int(AUDIO_DEVICE_INDEX)
    name = str(getattr(config, "AUDIO_DEVICE_NAME", "") or "").strip().lower()
    if not name:
        return None
    import sounddevice as sd
    for i, dev in enumerate(sd.query_devices()):
        if name in str(dev.get("name", "")).lower() and int(dev.get("max_input_channels", 0)) > 0:
            return i
    return None


def _record(secs: float, *, channels: int | None = None) -> np.ndarray:
    """Raw multi-channel capture, shape (n, ch). No gain, no channel selection."""
    import sounddevice as sd

    idx = _device_index()
    if idx is None:
        raise SystemExit(
            "No input device resolved. Set AUDIO_DEVICE_NAME or AUDIO_DEVICE_INDEX in .env"
        )
    info = sd.query_devices(idx)
    max_ch = int(info.get("max_input_channels", 1))
    ch = channels or max_ch or 1
    ch = max(1, min(ch, max_ch))
    frames = int(secs * SR)
    buf = sd.rec(frames, samplerate=SR, channels=ch, dtype="float32", device=idx)
    sd.wait()
    return np.asarray(buf, dtype=np.float32).reshape(frames, ch)


def _as_pipeline_mono(raw: np.ndarray) -> np.ndarray:
    """Apply exactly what audio/stream.py._callback does: channel select (or mix),
    then AUDIO_INPUT_GAIN with hard clipping. This is what Whisper sees."""
    ch_cfg = getattr(config, "AUDIO_AEC_INPUT_CHANNEL", -1)
    try:
        ch_cfg = int(ch_cfg)
    except (TypeError, ValueError):
        ch_cfg = -1
    n_ch = raw.shape[1]
    if n_ch > 1:
        mono = raw[:, ch_cfg].copy() if 0 <= ch_cfg < n_ch else raw.mean(axis=1).copy()
    else:
        mono = raw[:, 0].copy()
    gain = float(getattr(config, "AUDIO_INPUT_GAIN", 1.0) or 1.0)
    if gain != 1.0:
        mono = (mono * gain).clip(-1.0, 1.0)
    return mono


def _countdown(msg: str, secs: int = 3) -> None:
    print(f"\n{msg}")
    for i in range(secs, 0, -1):
        print(f"   starting in {i}...", end="\r", flush=True)
        time.sleep(1)
    print("   RECORDING            ")


def _config_banner() -> None:
    idx = _device_index()
    ch_cfg = getattr(config, "AUDIO_AEC_INPUT_CHANNEL", -1)
    gain = float(getattr(config, "AUDIO_INPUT_GAIN", 1.0) or 1.0)
    print("─" * 68)
    print("Live capture configuration (from .env / config.py)")
    print(f"  input device index      : {idx}")
    try:
        import sounddevice as sd
        info = sd.query_devices(idx)
        print(f"  device name             : {info.get('name')}")
        print(f"  max input channels      : {info.get('max_input_channels')}")
    except Exception as exc:
        print(f"  device query failed     : {exc}")
    print(f"  AUDIO_AEC_INPUT_CHANNEL : {ch_cfg!r}"
          f"{'  (blank/-1 => MIX all channels)' if str(ch_cfg).strip() in ('', '-1', 'None') else ''}")
    print(f"  AUDIO_INPUT_GAIN        : {gain}x")
    try:
        from audio import hardware_aec
        print(f"  hardware_aec.is_active(): {hardware_aec.is_active()}")
    except Exception as exc:
        print(f"  hardware_aec check      : {exc}")
    print("─" * 68)


# ── tests ─────────────────────────────────────────────────────────────────────

def test_channels(secs: float = 6.0) -> dict:
    """Identify what each physical mic channel carries.

    With the AEC firmware the ReSpeaker Lite puts echo-cancelled MIC audio on one
    channel and the raw playback REFERENCE on the other. If the reference channel
    is being mixed into the mono feed, the speech level is halved AND Rex's own
    output is added back — both of which wreck far-field transcription.
    """
    print("\n=== CHANNEL IDENTIFICATION ===")
    print("Talk normally from where you usually stand for the whole recording.")
    print("Do NOT play any audio through Rex during this test.")
    _countdown(f"Recording {secs:.0f}s of your voice...", 3)
    raw = _record(secs)
    n_ch = raw.shape[1]
    print(f"\nCaptured {n_ch} channel(s) at {SR} Hz\n")

    stats = []
    for c in range(n_ch):
        col = raw[:, c]
        stats.append({
            "ch": c,
            "rms_dbfs": _dbfs(col),
            "peak_dbfs": _peak_dbfs(col),
            "clip": _clip_fraction(col),
        })
        print(f"  channel {c}: RMS {stats[-1]['rms_dbfs']:7.1f} dBFS   "
              f"peak {stats[-1]['peak_dbfs']:7.1f} dBFS   "
              f"clipped {stats[-1]['clip'] * 100:5.2f}%")

    if n_ch >= 2:
        a, b = raw[:, 0], raw[:, 1]
        denom = (np.std(a) * np.std(b))
        corr = float(np.corrcoef(a, b)[0, 1]) if denom > 1e-9 else 0.0
        print(f"\n  channel 0/1 correlation: {corr:+.3f}")
        quiet = [s for s in stats if s["rms_dbfs"] < -70.0]
        print()
        if quiet:
            for s in quiet:
                print(f"  ! channel {s['ch']} is essentially SILENT ({s['rms_dbfs']:.1f} dBFS) "
                      f"while you were talking.")
            print("    That is the playback-reference channel (silent because Rex wasn't")
            print("    speaking). Mixing it in just halves your speech level.")
            live = [s for s in stats if s["rms_dbfs"] >= -70.0]
            if live:
                print(f"    -> set AUDIO_AEC_INPUT_CHANNEL={live[0]['ch']} in .env")
        elif abs(corr) > 0.9:
            print("  Both channels carry nearly the SAME signal (two mic capsules).")
            print("  Mixing them is fine; hardware AEC may not be flashed.")
        else:
            print("  Both channels carry DIFFERENT live audio — inspect before choosing.")
    return {"channels": stats}


def test_noise(secs: float = 5.0) -> float:
    """Measure the room noise floor through the live pipeline."""
    print("\n=== NOISE FLOOR ===")
    print("Stay SILENT. Leave the room's normal background (fans, TV, AC) running.")
    _countdown(f"Recording {secs:.0f}s of silence...", 3)
    mono = _as_pipeline_mono(_record(secs))
    floor = _dbfs(mono)
    print(f"\n  noise floor : {floor:6.1f} dBFS  (post-gain, as Whisper hears it)")
    print(f"  clipping    : {_clip_fraction(mono) * 100:5.2f}%")
    if floor > -45.0:
        print("  ! Loud room. Even a strong voice will struggle for SNR here.")
    return floor


def test_speech(secs: float = 8.0, floor_dbfs: float | None = None,
                label: str = "your normal spot") -> dict:
    """Measure speech level + SNR from wherever the user is standing."""
    print(f"\n=== SPEECH LEVEL ({label}) ===")
    print("Speak continuously at your normal conversational volume.")
    print('Suggested: count "one, two, three..." steadily until it stops.')
    _countdown(f"Recording {secs:.0f}s of speech...", 3)
    mono = _as_pipeline_mono(_record(secs))

    # Speech level = the loud half of the frames; noise = the quiet tenth. This
    # separates talking from the gaps between words without needing the VAD.
    win = int(0.03 * SR)
    frames = mono[: len(mono) // win * win].reshape(-1, win)
    if frames.size == 0:
        raise SystemExit("recording too short")
    rms = np.sqrt(np.mean(np.square(frames.astype(np.float64)), axis=1))
    loud = rms[rms >= np.percentile(rms, 50)]
    quiet = rms[rms <= np.percentile(rms, 10)]
    speech_db = 20.0 * np.log10(max(float(np.mean(loud)), 1e-6))
    gap_db = 20.0 * np.log10(max(float(np.mean(quiet)), 1e-6))
    ref_floor = floor_dbfs if floor_dbfs is not None else gap_db
    snr = speech_db - ref_floor
    clip = _clip_fraction(mono)

    print(f"\n  speech RMS  : {speech_db:6.1f} dBFS   (target ~{TARGET_SPEECH_DBFS:.0f})")
    print(f"  between-word: {gap_db:6.1f} dBFS")
    print(f"  SNR         : {snr:6.1f} dB      (good >= {GOOD_SNR_DB:.0f}, poor < {POOR_SNR_DB:.0f})")
    print(f"  peak        : {_peak_dbfs(mono):6.1f} dBFS")
    print(f"  clipping    : {clip * 100:5.2f}%")
    return {"speech_dbfs": speech_db, "snr_db": snr, "clip": clip, "audio": mono}


def test_distance() -> None:
    """Guided sweep so the distance dependence is measured, not assumed."""
    print("\n=== DISTANCE SWEEP ===")
    floor = test_noise()
    results = []
    for feet in (3, 6, 9):
        input(f"\nStand about {feet} ft from Rex, facing him, then press Enter...")
        res = test_speech(6.0, floor_dbfs=floor, label=f"{feet} ft")
        results.append((feet, res))
    print("\n" + "─" * 68)
    print(f"  {'distance':>10}  {'speech dBFS':>12}  {'SNR dB':>8}")
    for feet, res in results:
        print(f"  {feet:>8} ft  {res['speech_dbfs']:>12.1f}  {res['snr_db']:>8.1f}")
    print("─" * 68)
    print("Level should fall ~6 dB per doubling of distance. A much steeper drop")
    print("means the mic is being blocked/pointed away rather than simply far.")


def test_transcribe(secs: float = 8.0) -> None:
    """End-to-end: record at distance, then transcribe with the SAME backend Rex
    uses. This is the only test that shows the actual words he would have heard."""
    print("\n=== END-TO-END TRANSCRIPTION ===")
    print("Speak a known sentence from your normal spot, e.g.")
    print('  "Turn around and come forward five feet."')
    _countdown(f"Recording {secs:.0f}s...", 3)
    mono = _as_pipeline_mono(_record(secs))
    print(f"\n  level {_dbfs(mono):.1f} dBFS   clipping {_clip_fraction(mono) * 100:.2f}%")
    print("  transcribing (same backend as the robot)...")
    from audio import transcription
    text = transcription.transcribe(mono)
    print(f"\n  Rex would hear: {text!r}\n")


def _verdict(floor: float, speech: dict) -> None:
    snr, level, clip = speech["snr_db"], speech["speech_dbfs"], speech["clip"]
    gain = float(getattr(config, "AUDIO_INPUT_GAIN", 1.0) or 1.0)
    print("\n" + "=" * 68)
    print("VERDICT")
    print("=" * 68)

    if clip > CLIP_WARN_FRAC:
        print(f"  CLIPPING ({clip * 100:.1f}% of samples). Distortion hurts Whisper more")
        print(f"  than quiet audio does. Lower AUDIO_INPUT_GAIN (now {gain}x).")
    if snr >= GOOD_SNR_DB:
        print(f"  SNR {snr:.1f} dB is healthy. The mic is NOT your transcription problem —")
        print("  look at endpointing/VAD or the language model instead.")
    elif snr >= POOR_SNR_DB:
        print(f"  SNR {snr:.1f} dB is marginal. Expect occasional word errors, especially")
        print("  on short utterances and proper nouns.")
    else:
        print(f"  SNR {snr:.1f} dB is POOR — this alone explains bad transcription.")
        print("  Gain will NOT fix it (gain lifts speech and noise together).")
        print("  Fixes that actually raise SNR, in order of effect:")
        print("    1. Get the mic off/away from noise sources and fan-cooled surfaces.")
        print("    2. Confirm you are reading the AEC'd channel only (run: channels).")
        print("    3. ReSpeaker beamforming/AGC firmware, or move the mic closer.")

    if clip <= CLIP_WARN_FRAC and level < TARGET_SPEECH_DBFS - 6:
        head = TARGET_SPEECH_DBFS - level
        suggested = round(gain * (10 ** (head / 20.0)), 1)
        suggested = min(suggested, 4.0)
        print(f"\n  Speech sits {head:.0f} dB below target and is not clipping, so there is")
        print(f"  headroom: try AUDIO_INPUT_GAIN={suggested} (currently {gain}).")
        print("  Re-run afterwards — back off if clipping appears or startles return.")
    print("=" * 68)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("test", nargs="?", default="all",
                    choices=["channels", "noise", "speech", "distance", "transcribe", "all"])
    ap.add_argument("--secs", type=float, default=None, help="override recording length")
    args = ap.parse_args()

    _config_banner()
    try:
        if args.test == "channels":
            test_channels(args.secs or 6.0)
        elif args.test == "noise":
            test_noise(args.secs or 5.0)
        elif args.test == "speech":
            test_speech(args.secs or 8.0)
        elif args.test == "distance":
            test_distance()
        elif args.test == "transcribe":
            test_transcribe(args.secs or 8.0)
        else:
            test_channels(6.0)
            floor = test_noise(5.0)
            speech = test_speech(args.secs or 8.0, floor_dbfs=floor)
            _verdict(floor, speech)
    except KeyboardInterrupt:
        print("\ninterrupted")


if __name__ == "__main__":
    main()
