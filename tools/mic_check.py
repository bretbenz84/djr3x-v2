#!/usr/bin/env python3
"""
tools/mic_check.py — Far-field microphone diagnostic for the ReSpeaker
(Flex XVF3800 Circular-4 since 2026-09-02; the Lite before that).

Answers the question "why are transcriptions bad when I talk from across the
room?" with measurements instead of guesses. It reproduces the LIVE capture math
(audio/stream.py: channel selection, then AUDIO_INPUT_GAIN with hard clipping),
so what it reports is what Whisper actually receives.

    ./venv/bin/python tools/mic_check.py channels     # what is on each mic channel
    ./venv/bin/python tools/mic_check.py noise        # room noise floor (stay quiet)
    ./venv/bin/python tools/mic_check.py speech       # speak from your normal spot
    ./venv/bin/python tools/mic_check.py spectrum     # WHERE the noise is + filter payoff
    ./venv/bin/python tools/mic_check.py distance     # guided 3ft / 6ft / 9ft sweep
    ./venv/bin/python tools/mic_check.py transcribe   # end-to-end: hear it as Rex does
    ./venv/bin/python tools/mic_check.py listen       # record + PLAY BACK what Rex hears
    ./venv/bin/python tools/mic_check.py ab           # A/B one change (charger in/out)
    ./venv/bin/python tools/mic_check.py score        # scripted accuracy benchmark, logged
    ./venv/bin/python tools/mic_check.py aec          # PLAYS SOUND: echo cancellation per channel, logged
    ./venv/bin/python tools/mic_check.py all          # channels + noise + speech + verdict

Nothing here writes to the robot's databases. Only `aec` and `listen` make sound,
and both say so and wait for Enter first. Run it with the main app STOPPED (it
needs exclusive use of the mic).

Comparison records: `score` appends to logs/mic_check/history.jsonl and `aec` to
logs/mic_check/aec_history.jsonl, so a mic swap can be judged against the last
board's numbers instead of memory (the ReSpeaker Lite baselines live in
docs/respeaker_flex_xvf3800.md).

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


# Set by --device: capture through a DIFFERENT mic than the robot's .env one.
# The point is comparative diagnosis — run `score` on the ReSpeaker, then again
# with --device "MacBook" from the same spot: if another mic in the same room
# scores clearly better, the board (or its always-on DSP) is the weak link; if
# both score the same, the SNR is the room's physics and no board swap fixes it.
_DEVICE_OVERRIDE: "int | str | None" = None


def _device_index() -> int | None:
    if _DEVICE_OVERRIDE is not None:
        import sounddevice as sd
        try:
            return int(_DEVICE_OVERRIDE)
        except (TypeError, ValueError):
            pass
        want = str(_DEVICE_OVERRIDE).strip().lower()
        for i, dev in enumerate(sd.query_devices()):
            if want in str(dev.get("name", "")).lower() and int(dev.get("max_input_channels", 0)) > 0:
                return i
        raise SystemExit(f"--device {_DEVICE_OVERRIDE!r} matched no input device "
                         f"(try: python -c \"import sounddevice; print(sounddevice.query_devices())\")")
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

# Known USB channel layouts, keyed by channel count. Verified by capture +
# tools/flex_ctl.py on 2026-09-02 for the Flex; the Lite entry is the AEC
# firmware build that duplicated processed mono on both channels.
_KNOWN_LAYOUTS = {
    6: {
        "board": "reSpeaker Flex XVF3800 (6-ch USB build)",
        "roles": {0: "Conference output — AEC + beamform + NS + AGC (room floor ~26 dB hot)",
                  1: "ASR output of the auto-selected beam — AEC + beamform, no AGC",
                  2: "raw mic 0", 3: "raw mic 1", 4: "raw mic 2", 5: "raw mic 3"},
        "recommend": 1,
    },
}


def _correlation_clusters(corr: np.ndarray, threshold: float = 0.9) -> list[list[int]]:
    """Group channels whose pairwise |correlation| exceeds threshold (single-link)."""
    n = corr.shape[0]
    seen: set[int] = set()
    clusters: list[list[int]] = []
    for i in range(n):
        if i in seen:
            continue
        group = [i]
        seen.add(i)
        frontier = [i]
        while frontier:
            a = frontier.pop()
            for b in range(n):
                if b not in seen and abs(float(corr[a, b])) >= threshold:
                    seen.add(b)
                    group.append(b)
                    frontier.append(b)
        clusters.append(sorted(group))
    return clusters


def _classify_channels(stats: list[dict], corr: np.ndarray) -> dict:
    """Pure decision logic for test_channels, so it is unit-testable without a mic.

    Returns {"verdict": str, "recommend": int | None, "clusters": [...]}. recommend
    is the AUDIO_AEC_INPUT_CHANNEL to set (None = leave blank / mix).
    """
    n_ch = len(stats)
    clusters = _correlation_clusters(corr) if n_ch > 1 else [[0]]
    if n_ch in _KNOWN_LAYOUTS:
        layout = _KNOWN_LAYOUTS[n_ch]
        return {"verdict": f"known layout: {layout['board']}",
                "recommend": layout["recommend"], "clusters": clusters,
                "roles": layout["roles"]}
    if n_ch == 1:
        return {"verdict": "mono device", "recommend": None, "clusters": clusters}
    silent = [s["ch"] for s in stats if s["rms_dbfs"] < -70.0]
    live = [s["ch"] for s in stats if s["rms_dbfs"] >= -70.0]
    if silent and live:
        return {"verdict": f"channel(s) {silent} silent while you talked — playback reference; "
                           f"read the live channel", "recommend": live[0], "clusters": clusters}
    c01 = abs(float(corr[0, 1]))
    if n_ch == 2 and c01 > 0.999 and abs(stats[0]["rms_dbfs"] - stats[1]["rms_dbfs"]) < 0.5:
        return {"verdict": "two IDENTICAL channels — one processed mono stream duplicated; "
                           "mixing costs nothing", "recommend": None, "clusters": clusters}
    if n_ch == 2 and c01 > 0.9:
        return {"verdict": "two raw capsules (nearly the same signal); mixing is fine but no "
                           "array processing is applied", "recommend": None, "clusters": clusters}
    return {"verdict": "channels carry DIFFERENT audio — inspect before choosing",
            "recommend": None, "clusters": clusters}


def test_channels(secs: float = 6.0) -> dict:
    """Identify what each physical mic channel carries.

    Flex XVF3800 (6-ch build): 0 = Conference (AGC'd), 1 = ASR beam, 2-5 = raw
    capsules — channel 1 is the one our pipeline wants. Old ReSpeaker Lite AEC
    firmware: processed mono on both channels. Mixing the wrong channels either
    halves speech level, adds Rex's own output back, or (Flex) buries everything
    under the AGC channel's 26 dB-hot noise floor.
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

    corr = np.eye(n_ch)
    if n_ch >= 2:
        stds = raw.std(axis=0)
        if np.all(stds > 1e-9):
            corr = np.corrcoef(raw.T)
        print("\n  correlation matrix:")
        for i in range(n_ch):
            print("   ", " ".join(f"{float(corr[i, j]):+.2f}" for j in range(n_ch)))

    result = _classify_channels(stats, corr)
    print(f"\n  {result['verdict']}")
    for group in result["clusters"]:
        if len(group) > 1:
            print(f"    channels {group} move together")
    for ch, role in (result.get("roles") or {}).items():
        print(f"    ch{ch}: {role}")
    if result["recommend"] is None:
        print("  -> leave AUDIO_AEC_INPUT_CHANNEL blank (mix).")
    else:
        print(f"  -> set AUDIO_AEC_INPUT_CHANNEL={result['recommend']} in .env")
    cur = getattr(config, "AUDIO_AEC_INPUT_CHANNEL", -1)
    try:
        cur = int(cur)
    except (TypeError, ValueError):
        cur = -1
    if (cur if cur >= 0 else None) != result["recommend"]:
        print(f"  ! .env currently has {cur if cur >= 0 else 'blank'} — change it.")
    return {"channels": stats, "classification": result}


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


def _band_energy(x: np.ndarray, edges: list[tuple[float, float]]) -> list[float]:
    """RMS dBFS per frequency band, via a real FFT magnitude spectrum."""
    if x.size < 1024:
        return [-120.0] * len(edges)
    spec = np.abs(np.fft.rfft(x.astype(np.float64) * np.hanning(x.size)))
    freqs = np.fft.rfftfreq(x.size, 1.0 / SR)
    # Parseval-consistent scaling so band sums are comparable to a broadband RMS.
    power = (spec ** 2) / (x.size * np.sum(np.hanning(x.size) ** 2) / x.size)
    out = []
    for lo, hi in edges:
        sel = (freqs >= lo) & (freqs < hi)
        p = float(np.sum(power[sel])) * 2.0 / (x.size ** 2) * x.size
        out.append(10.0 * np.log10(p) if p > 1e-12 else -120.0)
    return out


def test_spectrum(secs: float = 5.0) -> None:
    """Locate the noise in FREQUENCY, and price a high-pass filter.

    Makeup gain cannot improve SNR, but removing noise where speech ISN'T can.
    Room rumble (HVAC, fans, servo whine, structural) piles up below ~150 Hz,
    while speech intelligibility for ASR lives roughly 150 Hz - 6 kHz. If the
    noise floor is bottom-heavy, a high-pass is free SNR.
    """
    print("\n=== NOISE SPECTRUM ===")
    print("Stay SILENT — measuring where the room noise actually sits.")
    _countdown(f"Recording {secs:.0f}s of silence...", 3)
    noise = _as_pipeline_mono(_record(secs))

    print("\nSpeak normally from your usual spot for the comparison.")
    _countdown(f"Recording {secs:.0f}s of speech...", 3)
    speech = _as_pipeline_mono(_record(secs))

    bands = [(0, 80), (80, 150), (150, 300), (300, 1000),
             (1000, 3000), (3000, 6000), (6000, 8000)]
    n_db = _band_energy(noise, bands)
    s_db = _band_energy(speech, bands)

    print(f"\n  {'band (Hz)':>12}  {'noise dB':>9}  {'speech dB':>10}  {'SNR dB':>7}")
    for (lo, hi), nd, sd in zip(bands, n_db, s_db):
        print(f"  {f'{lo}-{hi}':>12}  {nd:>9.1f}  {sd:>10.1f}  {sd - nd:>7.1f}")

    # Price a high-pass at each candidate cutoff: how much noise power is removed
    # versus how much speech power is sacrificed.
    print("\n  If a high-pass filter were applied to the capture:")
    total_n = sum(10 ** (d / 10.0) for d in n_db)
    total_s = sum(10 ** (d / 10.0) for d in s_db)
    base_snr = 10.0 * np.log10(total_s / total_n) if total_n > 0 else 0.0
    print(f"    {'cutoff':>8}  {'noise cut':>10}  {'speech lost':>12}  {'net SNR gain':>13}")
    for cut_idx, cutoff in ((1, 80), (2, 150), (3, 300)):
        kept_n = sum(10 ** (d / 10.0) for d in n_db[cut_idx:])
        kept_s = sum(10 ** (d / 10.0) for d in s_db[cut_idx:])
        if kept_n <= 0 or kept_s <= 0:
            continue
        new_snr = 10.0 * np.log10(kept_s / kept_n)
        print(f"    {cutoff:>6} Hz  "
              f"{10 * np.log10(total_n / kept_n):>9.1f} dB  "
              f"{10 * np.log10(total_s / kept_s):>11.1f} dB  "
              f"{new_snr - base_snr:>12.1f} dB")
    print(f"\n  (current broadband SNR over these bands: {base_snr:.1f} dB)")
    print("  A net gain of >= 2 dB is worth filtering; below that the noise is")
    print("  broadband (it sits ON the speech) and only distance/room fixes help.")


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


def _playback(mono: np.ndarray, label: str, boost_db: float = 0.0,
              normalize_to: float | None = None) -> None:
    """Play a captured take through the default output so you hear WHAT REX HEARS.
    boost_db lifts very quiet material (room noise floor) into the audible range;
    normalize_to plays SPEECH level-matched to a comfortable dBFS — far-field
    speech sits ~-36 dBFS and verbatim playback of that is nearly inaudible next
    to mastered audio, which reads as 'the mic is broken' when it isn't. The
    print always says when the level is not verbatim."""
    import sounddevice as sd

    x = mono
    if normalize_to is not None:
        cur = _dbfs(mono)
        lift = max(0.0, normalize_to - cur)
        x = np.clip(mono * (10.0 ** (lift / 20.0)), -1.0, 1.0)
        print(f"  ▶ playback ({label}, level-matched for listening: raw {cur:.1f} dBFS "
              f"lifted +{lift:.0f} dB — Whisper gets the RAW level)...")
    elif boost_db > 0.0:
        x = np.clip(mono * (10.0 ** (boost_db / 20.0)), -1.0, 1.0)
        print(f"  ▶ playback ({label}, boosted +{boost_db:.0f} dB so it's audible)...")
    else:
        print(f"  ▶ playback ({label}, verbatim level)...")
    try:
        sd.play(x, SR)
        sd.wait()
    except Exception as exc:
        print(f"  playback failed: {exc}")


def test_listen(secs: float = 6.0) -> None:
    """Record through the exact robot capture path, then play it back — first
    verbatim, then boosted. Use it two ways: stay silent to HEAR the room noise
    Rex sits in, or speak from your normal spot to hear yourself as he does."""
    print("\n=== LISTEN: HEAR WHAT REX HEARS ===")
    print("Stay silent to audition the room noise, or speak from your normal spot.")
    _countdown(f"Recording {secs:.0f}s...", 3)
    mono = _as_pipeline_mono(_record(secs))
    outdir = _ROOT / "logs" / "mic_check"
    from datetime import datetime
    path = outdir / f"listen-{datetime.now().strftime('%Y%m%d-%H%M%S')}.wav"
    _save_wav(path, mono)
    print(f"  level {_dbfs(mono):.1f} dBFS   clipping {_clip_fraction(mono) * 100:.2f}%"
          + "".join(f"   {hz}Hz hum +{db:.0f}dB" for hz, db in _hum_db(mono) if db >= 10.0))
    if _dbfs(mono) < -45.0:
        _playback(mono, "as captured")
        _playback(mono, "noise floor", boost_db=30.0)
    else:
        _playback(mono, "as captured")
        _playback(mono, "level-matched", normalize_to=-20.0)
    print(f"  saved: {path}")


def _save_wav(path: Path, mono: np.ndarray) -> None:
    """Write pipeline-mono float audio as 16-bit WAV (stdlib only)."""
    import wave

    path.parent.mkdir(parents=True, exist_ok=True)
    pcm = (np.clip(mono, -1.0, 1.0) * 32767.0).astype(np.int16)
    with wave.open(str(path), "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(SR)
        w.writeframes(pcm.tobytes())


def _hum_db(x: np.ndarray) -> list[tuple[int, float]]:
    """Mains-hum check: for each of 60/120/180/240 Hz, how many dB the spectrum
    peak within ±3 Hz sits above the local floor (median of ±5–25 Hz around it).
    Values above ~10 dB mean a tonal hum component is really there — the signature
    of a charger/supply ground loop rather than broadband room noise."""
    spec = np.abs(np.fft.rfft(x * np.hanning(x.size)))
    freqs = np.fft.rfftfreq(x.size, 1.0 / SR)
    out = []
    for hz in (60, 120, 180, 240):
        band = spec[(freqs >= hz - 3) & (freqs <= hz + 3)]
        ring = spec[((freqs >= hz - 25) & (freqs <= hz - 5))
                    | ((freqs >= hz + 5) & (freqs <= hz + 25))]
        if band.size == 0 or ring.size == 0:
            continue
        floor = float(np.median(ring)) or 1e-12
        out.append((hz, 20.0 * np.log10(float(np.max(band)) / floor)))
    return out


_AB_BANDS = [(0.0, 120.0), (120.0, 300.0), (300.0, 1000.0),
             (1000.0, 3000.0), (3000.0, 8000.0)]


def _ab_condition(label: str, secs: float, outdir: Path) -> dict:
    input(f"\n[{label}] Set up this condition, keep the room QUIET, press Enter...")
    _countdown(f"[{label}] Recording {secs:.0f}s of room noise...", 3)
    mono = _as_pipeline_mono(_record(secs))
    _save_wav(outdir / f"ab-{label}.wav", mono)
    bands = _band_energy(mono, _AB_BANDS)
    hum = _hum_db(mono)
    print(f"  [{label}] floor {_dbfs(mono):.1f} dBFS"
          + "".join(f"   {hz}Hz +{db:.0f}dB" for hz, db in hum if db >= 6.0))
    _playback(mono, f"condition {label} room noise", boost_db=30.0)
    return {"floor": _dbfs(mono), "bands": bands, "hum": dict(hum)}


def test_ab(secs: float = 6.0) -> None:
    """Two identical noise measurements around ONE deliberate change — built to
    settle 'what changed since it worked'. Prime suspect from the 2026-07-31 run:
    the battery charger (Rex was deaf from boot until 3 s after it was unplugged).
    Use it for any single variable: charger, AC, a moved mic, a new appliance."""
    print("\n=== A/B NOISE COMPARISON ===")
    print("Measure the same thing twice, changing exactly ONE condition between")
    print("runs — e.g. A = charger PLUGGED IN, B = charger UNPLUGGED.")
    outdir = _ROOT / "logs" / "mic_check"
    a = _ab_condition("A", secs, outdir)
    print("\nNow change the ONE thing (e.g. unplug the charger).")
    b = _ab_condition("B", secs, outdir)

    print("\n" + "─" * 68)
    print(f"  {'':>14}  {'A':>10}  {'B':>10}  {'delta':>8}")
    print(f"  {'floor dBFS':>14}  {a['floor']:>10.1f}  {b['floor']:>10.1f}"
          f"  {a['floor'] - b['floor']:>+8.1f}")
    for (lo, hi), ea, eb in zip(_AB_BANDS, a["bands"], b["bands"]):
        da = 10.0 * np.log10(max(ea, 1e-12) / max(eb, 1e-12))
        print(f"  {f'{lo:.0f}-{hi:.0f} Hz':>14}  {'':>10}  {'':>10}  {da:>+8.1f}")
    print("─" * 68)
    delta = a["floor"] - b["floor"]
    if delta >= 3.0:
        print(f"  Condition A is {delta:.1f} dB NOISIER than B — the changed thing is a")
        print("  real noise source. Every dB of floor is a dB of SNR Whisper loses.")
    elif delta <= -3.0:
        print(f"  Condition B is {-delta:.1f} dB noisier than A.")
    else:
        print("  No meaningful floor difference — this variable is NOT the problem.")
    worst_hum = max(list(a["hum"].values()) + [0.0])
    if worst_hum >= 10.0 and worst_hum > max(list(b["hum"].values()) + [0.0]) + 6.0:
        print("  A shows tonal mains hum that B lacks — classic charger/ground-loop")
        print("  signature. Keep that source unplugged (or on another circuit) while")
        print("  Rex listens.")
    print(f"  Recordings saved under {outdir}/ — listen to them.")


def _resolve_output_index():
    """The playback device the robot uses (config.AUDIO_OUTPUT_DEVICE_*), else None."""
    import sounddevice as sd
    idx = int(getattr(config, "AUDIO_OUTPUT_DEVICE_INDEX", -1) or -1)
    if idx >= 0:
        return idx
    name = str(getattr(config, "AUDIO_OUTPUT_DEVICE_NAME", "") or "").strip().lower()
    if not name:
        return None
    for i, dev in enumerate(sd.query_devices()):
        if name in str(dev.get("name", "")).lower() and int(dev.get("max_output_channels", 0)) > 0:
            return i
    return None


def _rex_speech_program(secs: float) -> np.ndarray:
    """secs of Rex's own cached voice at 16 kHz, peak-normalized to -12 dBFS.

    His real lines are the echo the AEC has to cancel in service, so that is what
    we play. Falls back to a shaped 1 kHz tone when the TTS cache is empty.
    """
    import glob
    import math

    parts: list[np.ndarray] = []
    total = 0
    cache = sorted(glob.glob(str(_ROOT / "assets" / "audio" / "tts_cache" / "*.wav")),
                   key=lambda p: Path(p).stat().st_size, reverse=True)
    try:
        import soundfile as sf
        from scipy.signal import resample_poly
    except Exception:
        cache = []
    for path in cache:
        try:
            a, sr = sf.read(path, dtype="float32", always_2d=True)
        except Exception:
            continue
        a = a.mean(axis=1)
        if sr != SR:
            g = math.gcd(int(sr), SR)
            a = resample_poly(a, SR // g, int(sr) // g).astype(np.float32)
        parts.append(a)
        total += len(a)
        if total >= secs * SR:
            break
    if parts:
        x = np.concatenate(parts)[: int(secs * SR)]
    else:
        t = np.arange(int(secs * SR)) / SR
        x = np.sin(2 * np.pi * 1000.0 * t).astype(np.float32)
    x = x / max(1e-6, float(np.max(np.abs(x)))) * (10.0 ** (-12.0 / 20.0))
    r = int(0.05 * SR)
    x[:r] *= np.linspace(0.0, 1.0, r)
    x[-r:] *= np.linspace(1.0, 0.0, r)
    return x.astype(np.float32)


def _erle_windows(rec: np.ndarray, processed: list[int], raw: list[int],
                  win_secs: float = 1.0) -> list[dict]:
    """Per-window echo return loss enhancement: raw-capsule level minus each
    processed channel's level, in dB. Pure math, unit-tested."""
    win = max(1, int(win_secs * SR))
    out = []
    for start in range(0, len(rec) - win + 1, win):
        w = rec[start:start + win]
        raw_db = float(np.mean([_dbfs(w[:, c]) for c in raw])) if raw else float("nan")
        row = {"t": start / SR, "raw_dbfs": raw_db}
        for c in processed:
            lvl = _dbfs(w[:, c])
            row[f"ch{c}_dbfs"] = lvl
            row[f"erle_ch{c}"] = raw_db - lvl if raw else float("nan")
        out.append(row)
    return out


def test_aec(secs: float = 20.0) -> dict | None:
    """Measure the on-chip echo cancellation: play Rex's own voice OUT through the
    robot's speaker and record every mic channel at the same time.

    Reports, per processed channel, how far below the raw capsules the echo lands
    (ERLE) window by window — the adaptive filter needs a few seconds of far-end
    audio to converge, so the number that matters is the LAST few seconds, not
    the first. The XVF3800's own AEC_AECCONVERGED flag is read afterwards when
    tools/flex_ctl.py can reach the board. Appended to logs/mic_check/aec_history.jsonl.

    THIS TEST MAKES SOUND. It says so and waits for Enter.
    """
    import json
    import sounddevice as sd
    from datetime import datetime

    in_idx = _device_index()
    out_idx = _resolve_output_index()
    if in_idx is None:
        raise SystemExit("no input device resolved (AUDIO_DEVICE_NAME / AUDIO_DEVICE_INDEX)")
    if out_idx is None:
        raise SystemExit("no output device resolved — set AUDIO_OUTPUT_DEVICE_NAME to the "
                         "ReSpeaker so the chip gets its echo reference")
    info_in = sd.query_devices(in_idx)
    info_out = sd.query_devices(out_idx)
    n_ch = int(info_in.get("max_input_channels", 1))
    layout = _KNOWN_LAYOUTS.get(n_ch)
    processed = [0, 1] if n_ch >= 2 else [0]
    raw = [c for c in range(2, n_ch)] if layout else []

    print("\n=== HARDWARE AEC (ECHO CANCELLATION) ===")
    print(f"  will PLAY {secs:.0f}s of Rex's own cached voice, peak -12 dBFS, through")
    print(f"  output device {out_idx} ({info_out.get('name')}) — the robot's speaker —")
    print(f"  while recording all {n_ch} channel(s) of input device {in_idx} ({info_in.get('name')}).")
    if not raw:
        print("  (no raw-capsule channels on this device, so only level RISE per channel")
        print("   can be reported, not a true ERLE)")
    print("  Keep the room quiet (TV off) and do not talk during the recording.")
    if input("\n  Press Enter to play, or Ctrl-C to abort... ") is None:
        return None

    program = _rex_speech_program(secs)
    out = np.stack([program, program], axis=1)
    quiet = _record(2.0)
    rec = sd.playrec(out, samplerate=SR, channels=n_ch, dtype="float32",
                     device=(in_idx, out_idx))
    sd.wait()
    rec = np.asarray(rec, dtype=np.float32).reshape(len(program), n_ch)

    rows = _erle_windows(rec, processed, raw, 1.0)
    print(f"\n  {'t':>4}  {'raw':>7}  " + "  ".join(f"{'ch' + str(c):>7}  {'ERLE' + str(c):>6}" for c in processed))
    for r in rows:
        print(f"  {r['t']:4.0f}  {r['raw_dbfs']:7.1f}  "
              + "  ".join(f"{r[f'ch{c}_dbfs']:7.1f}  {r[f'erle_ch{c}']:6.1f}" for c in processed))
    tail = rows[-5:] if len(rows) >= 5 else rows
    summary = {}
    for c in processed:
        summary[f"erle_ch{c}_last5s"] = float(np.mean([r[f"erle_ch{c}"] for r in tail]))
        summary[f"ch{c}_quiet_dbfs"] = _dbfs(quiet[:, c])
        summary[f"ch{c}_residual_dbfs"] = float(np.mean([r[f"ch{c}_dbfs"] for r in tail]))
    if raw:
        summary["raw_echo_dbfs"] = float(np.mean([r["raw_dbfs"] for r in tail]))
        summary["raw_peak_dbfs"] = float(max(_peak_dbfs(rec[:, c]) for c in raw))

    print("\n  last 5 s:")
    for c in processed:
        role = (layout or {}).get("roles", {}).get(c, "")
        print(f"    ch{c}: residual {summary[f'ch{c}_residual_dbfs']:6.1f} dBFS "
              f"(quiet floor {summary[f'ch{c}_quiet_dbfs']:6.1f})  "
              f"ERLE {summary[f'erle_ch{c}_last5s']:5.1f} dB   {role}")
    if raw:
        print(f"    raw capsules: echo {summary['raw_echo_dbfs']:.1f} dBFS, "
              f"peak {summary['raw_peak_dbfs']:.1f} dBFS"
              + ("  ! raw mics near clipping — lower the amp" if summary["raw_peak_dbfs"] > -3.0 else ""))

    converged = None
    dsp: dict = {}
    try:
        from tools import flex_ctl
        dsp = flex_ctl.snapshot(["AEC_AECCONVERGED", "AEC_AECPATHCHANGE", "AEC_RT60",
                                 "AUDIO_MGR_SYS_DELAY", "AUDIO_MGR_MIC_GAIN",
                                 "AUDIO_MGR_REF_GAIN", "PP_AGCONOFF", "PP_ECHOONOFF",
                                 "AEC_HPFONOFF", "VERSION", "BLD_MSG"])
        conv = dsp.get("AEC_AECCONVERGED")
        if isinstance(conv, list):
            converged = bool(conv[0])
            print(f"    XVF3800 reports AEC converged = {converged}")
    except Exception as exc:
        print(f"    (flex_ctl unavailable: {exc})")

    # Verdict against the numbers that matter for barge-in / talk-over.
    sel = getattr(config, "AUDIO_AEC_INPUT_CHANNEL", -1)
    try:
        sel = int(sel)
    except (TypeError, ValueError):
        sel = -1
    key = f"erle_ch{sel}_last5s" if sel in processed else f"erle_ch{processed[-1]}_last5s"
    erle = summary[key]
    print()
    if raw:
        if erle >= 25.0:
            print(f"  {erle:.0f} dB on the pipeline channel: strong — wake word over Rex's voice is realistic.")
        elif erle >= 15.0:
            print(f"  {erle:.0f} dB on the pipeline channel: on par with the old ReSpeaker Lite (~17 dB).")
        elif erle >= 6.0:
            print(f"  {erle:.0f} dB: weak. Check the filter converged (re-run longer), that the amp")
            print("  is on the Flex's OWN output, and that AUDIO_OUTPUT_DEVICE_NAME routes here.")
        else:
            print(f"  {erle:.0f} dB: NOT cancelling. The chip is not seeing this playback as its")
            print("  reference (wrong output device) or the AEC is bypassed (SHF_BYPASS).")
    print()

    outdir = _ROOT / "logs" / "mic_check"
    stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    outdir.mkdir(parents=True, exist_ok=True)
    for c in processed:
        _save_wav(outdir / f"aec-{stamp}-ch{c}.wav", rec[:, c])
    if raw:
        _save_wav(outdir / f"aec-{stamp}-raw{raw[0]}.wav", rec[:, raw[0]])
    record = {"ts": stamp, "device": info_in.get("name"), "output": info_out.get("name"),
              "secs": secs, "channels": n_ch, "processed": processed, "raw": raw,
              "pipeline_channel": sel, "summary": summary, "converged": converged,
              "dsp": {k: v for k, v in dsp.items() if not k.startswith("_")},
              "windows": rows}
    with (outdir / "aec_history.jsonl").open("a") as f:
        f.write(json.dumps(record) + "\n")
    print(f"  logged to {outdir / 'aec_history.jsonl'}; residual takes saved as aec-{stamp}-*.wav")
    return record


# Mixed lengths on purpose: the 1-2 word lines are Rex's known weak spot (short
# utterances give Whisper almost no context and get mislabeled/homophoned in the
# field), the long ones test sustained accuracy. Keep both ends represented.
_SCORE_SENTENCES = [
    "Halt.",
    "Come here.",
    "Rex, can you hear me?",
    "Turn around and come forward five feet.",
    "I'm going to watch a movie tonight.",
    "What do you see in this room right now?",
    "Play some cantina music and lower the volume.",
    "When you get a chance, could you tell me what the weather is going to be like tomorrow afternoon in Sacramento?",
]


def _word_accuracy(ref: str, hyp: str) -> float:
    """1 - WER, floored at 0. Tokens are lowercased with punctuation stripped."""
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


def test_score(secs: float = 6.0) -> None:
    """Scripted end-to-end transcription benchmark through the EXACT robot path
    (pipeline mono + the robot's transcription backend), scored as word accuracy
    and appended to logs/mic_check/history.jsonl — so 'it used to be rock solid'
    becomes a number you can compare across days, positions, and fixes. Stand at
    your NORMAL talking spot and read each sentence naturally."""
    import json
    from datetime import datetime

    print("\n=== SCORED TRANSCRIPTION BENCHMARK ===")
    from audio import transcription
    outdir = _ROOT / "logs" / "mic_check"
    stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    floor = test_noise(4.0)
    rows = []
    for i, ref in enumerate(_SCORE_SENTENCES, 1):
        # Record the INSTANT Enter lands — a countdown invites starting the
        # sentence early, and a clipped head turns "Rex, can you hear me?" into
        # "Ready?" and torpedoes the score (field 2026-07-31, take 1: 0%).
        input(f'\n[{i}/{len(_SCORE_SENTENCES)}] Get ready to read: "{ref}"\n'
              f'    Press Enter, then speak immediately...')
        print("   RECORDING — speak now")
        mono = _as_pipeline_mono(_record(secs))
        _save_wav(outdir / f"score-{stamp}-{i}.wav", mono)
        hyp = str(transcription.transcribe(mono) or "").strip()
        acc = _word_accuracy(ref, hyp)
        rows.append({"ref": ref, "heard": hyp, "accuracy": round(acc, 3),
                     "dbfs": round(_dbfs(mono), 1)})
        print(f"   heard: {hyp!r}   accuracy {acc * 100:.0f}%")
        _playback(mono, "what Rex heard", normalize_to=-20.0)
    mean_acc = float(np.mean([r["accuracy"] for r in rows]))
    print("\n" + "─" * 68)
    for r in rows:
        print(f"  {r['accuracy'] * 100:>4.0f}%  {r['dbfs']:>7.1f} dBFS  {r['heard']!r}")
    print("─" * 68)
    print(f"  MEAN WORD ACCURACY: {mean_acc * 100:.0f}%   (noise floor {floor:.1f} dBFS)")
    print("  >=95% healthy | 85-95% marginal | <85% transcription is genuinely broken")
    try:
        import sounddevice as sd
        dev_name = str(sd.query_devices(_device_index()).get("name") or "?")
    except Exception:
        dev_name = "?"
    record = {"ts": stamp, "device": dev_name,
              "mean_accuracy": round(mean_acc, 3), "floor_dbfs": round(floor, 1),
              "gain": float(getattr(config, "AUDIO_INPUT_GAIN", 1.0) or 1.0),
              "rows": rows}
    hist = outdir / "history.jsonl"
    hist.parent.mkdir(parents=True, exist_ok=True)
    with hist.open("a") as f:
        f.write(json.dumps(record) + "\n")
    print(f"  Appended to {hist} — re-run after any change and compare.")


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
        suggested = min(round(gain * (10 ** (head / 20.0)), 1), 4.0)
        print(f"\n  Speech sits {head:.0f} dB below target with no clipping, so there IS")
        print(f"  level headroom (AUDIO_INPUT_GAIN={suggested} vs {gain} today).")
        if snr < GOOD_SNR_DB:
            # Be explicit: makeup gain is a multiply — it raises the noise by the
            # same dB it raises the voice. It cannot buy back SNR, and this is the
            # exact trap the .env history records (6x and 2x both "ran hot").
            print("  BUT gain multiplies speech AND noise equally — it will NOT improve")
            print(f"  the {snr:.0f} dB SNR above, so it will not fix word errors. Raise it")
            print("  only if quiet audio is causing MISSED speech (VAD not triggering),")
            print("  not to chase accuracy. Fix the SNR first.")
        else:
            print("  SNR is healthy, so this is a safe level-only lift.")
        print("  Re-run afterwards — back off if clipping appears or startles return.")
    print("=" * 68)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("test", nargs="?", default="all",
                    choices=["channels", "noise", "speech", "spectrum",
                             "distance", "transcribe", "listen", "ab", "score", "aec", "all"])
    ap.add_argument("--secs", type=float, default=None, help="override recording length")
    ap.add_argument("--device", default=None,
                    help="capture through a different input device (name substring or "
                         "index) instead of the robot's .env mic — for A/B'ing the "
                         "ReSpeaker against e.g. the MacBook mic from the same spot")
    args = ap.parse_args()

    global _DEVICE_OVERRIDE
    _DEVICE_OVERRIDE = args.device

    _config_banner()
    try:
        if args.test == "channels":
            test_channels(args.secs or 6.0)
        elif args.test == "listen":
            test_listen(args.secs or 6.0)
        elif args.test == "ab":
            test_ab(args.secs or 6.0)
        elif args.test == "score":
            test_score(args.secs or 6.0)
        elif args.test == "aec":
            test_aec(args.secs or 20.0)
        elif args.test == "noise":
            test_noise(args.secs or 5.0)
        elif args.test == "speech":
            test_speech(args.secs or 8.0)
        elif args.test == "spectrum":
            test_spectrum(args.secs or 5.0)
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
