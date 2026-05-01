#!/usr/bin/env python3
"""
tools/audio_test.py — Standalone diagnostic for the DJ-R3X v2 audio pipeline.

Imports from the project but does not start the full system.
Each test is self-contained and exits cleanly on Ctrl+C.

Usage:
    python tools/audio_test.py --test mic
    python tools/audio_test.py --test vad
    python tools/audio_test.py --test aec
    python tools/audio_test.py --test transcription
    python tools/audio_test.py --test pipeline
"""

import argparse
import os
import sys
import threading
import time
from pathlib import Path

# ── Project root on sys.path ───────────────────────────────────────────────────
_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_ROOT))
os.chdir(_ROOT)

# Load .env before importing anything that reads hardware config.
# This must happen before any project import that touches audio device config.
try:
    from dotenv import load_dotenv
    load_dotenv(_ROOT / ".env")
except ImportError:
    pass  # dotenv missing; env vars must be set in the shell

try:
    import numpy as np
except ImportError:
    print("ERROR: numpy not installed. Run: pip install numpy")
    sys.exit(1)


# ── Helpers ────────────────────────────────────────────────────────────────────

def ts() -> str:
    ms = int(time.time() * 1000) % 1000
    return f"[{time.strftime('%H:%M:%S')}.{ms:03d}]"


def _bar(value: float, width: int = 30) -> str:
    n = int(min(1.0, max(0.0, value)) * width)
    return "█" * n + "░" * (width - n)


def _die(msg: str) -> None:
    print(f"\n{ts()} ERROR: {msg}", flush=True)
    sys.exit(1)


def _get_device_index() -> "int | None":
    name = os.getenv("AUDIO_DEVICE_NAME", "").strip()
    if name:
        resolved = _resolve_device_name(name)
        if resolved is not None:
            return resolved
    raw = os.getenv("AUDIO_DEVICE_INDEX", "").strip()
    if not raw:
        return None
    try:
        return int(raw)
    except ValueError:
        return None


def _resolve_device_name(name: str) -> "int | None":
    try:
        import sounddevice as sd
        devices = list(sd.query_devices())
    except Exception:
        return None
    wanted = name.lower()
    inputs = [
        (i, str(d.get("name") or ""))
        for i, d in enumerate(devices)
        if int(d.get("max_input_channels", 0) or 0) > 0
    ]
    exact = [(i, n) for i, n in inputs if n.lower() == wanted]
    if exact:
        return exact[0][0]
    contains = [(i, n) for i, n in inputs if wanted in n.lower()]
    if len(contains) == 1:
        return contains[0][0]
    return None


def _list_input_devices() -> None:
    try:
        import sounddevice as sd
        idx = _get_device_index()
        configured_name = os.getenv("AUDIO_DEVICE_NAME", "").strip()
        print(f"{ts()} Available input devices:")
        for i, d in enumerate(sd.query_devices()):
            if d.get("max_input_channels", 0) > 0:
                marker = "  ← configured mic"
                marker = marker if i == idx else ""
                print(f"  {i:2d}: {d['name']} (in: {d['max_input_channels']}){marker}")
        if configured_name and idx is None:
            print(f"{ts()} AUDIO_DEVICE_NAME={configured_name!r} did not resolve to a unique input device.")
    except Exception as exc:
        print(f"{ts()} Could not query devices: {exc}")


def _check_sounddevice() -> None:
    try:
        import sounddevice  # noqa: F401
    except ImportError:
        _die("sounddevice not installed. Run: pip install sounddevice")


def _load_silero() -> "tuple[object, bool]":
    """Load Silero VAD model. Returns (model, ok)."""
    try:
        import torch
        from silero_vad import load_silero_vad
        model = load_silero_vad()
        return model, True
    except ImportError:
        print(f"{ts()} WARNING: silero-vad or torch not installed — VAD unavailable.")
        print(f"           Run: pip install silero-vad torch")
        return None, False
    except Exception as exc:
        print(f"{ts()} WARNING: Silero VAD failed to load: {exc}")
        return None, False


def _silero_prob(model, chunk: np.ndarray, sample_rate: int) -> float:
    """Return raw Silero speech probability for one chunk."""
    try:
        import torch
        tensor = torch.from_numpy(chunk.astype(np.float32))
        with torch.no_grad():
            return float(model(tensor, sample_rate).item())
    except Exception:
        return 0.0


# ─────────────────────────────────────────────────────────────────────────────
# --test mic
# ─────────────────────────────────────────────────────────────────────────────

def test_mic() -> None:
    """Real-time mic level meter."""
    _check_sounddevice()
    import sounddevice as sd

    try:
        import config
        sample_rate = config.AUDIO_SAMPLE_RATE
    except Exception:
        sample_rate = 16000

    device_idx = _get_device_index()

    print(f"{ts()} Mic test")
    _list_input_devices()
    print()

    if device_idx is None:
        print(f"{ts()} ERROR: AUDIO_DEVICE_NAME/AUDIO_DEVICE_INDEX not set or not found in .env")
        print(f"         Set AUDIO_DEVICE_NAME to your microphone name, or AUDIO_DEVICE_INDEX as a fallback.")
        sys.exit(1)

    print(f"{ts()} Device: {device_idx}  Sample rate: {sample_rate} Hz")
    print(f"{ts()} Press Ctrl+C to stop.")
    print()

    blocksize = int(sample_rate * 0.032)
    peak = 0.0

    def callback(indata, frames, t, status):
        nonlocal peak
        rms = float(np.sqrt(np.mean(indata[:, 0] ** 2)))
        peak = max(peak, rms)
        bar = _bar(rms / 0.25, 40)
        sys.stdout.write(f"\r{ts()} RMS: {bar} {rms:.4f}  peak: {peak:.4f}  ")
        sys.stdout.flush()

    try:
        stream = sd.InputStream(
            device=device_idx,
            samplerate=sample_rate,
            channels=1,
            dtype="float32",
            blocksize=blocksize,
            callback=callback,
        )
        with stream:
            stream.start()
            while True:
                time.sleep(0.1)
    except KeyboardInterrupt:
        print(f"\n{ts()} Stopped.")
    except sd.PortAudioError as exc:
        print()
        _die(
            f"PortAudio error opening device {device_idx}: {exc}\n"
            f"  → Check that AUDIO_DEVICE_NAME/AUDIO_DEVICE_INDEX resolves to the correct connected mic.\n"
            f"  → Re-run to see the full device list above."
        )


# ─────────────────────────────────────────────────────────────────────────────
# --test vad
# ─────────────────────────────────────────────────────────────────────────────

def test_vad() -> None:
    """Real-time VAD decisions with Silero confidence score."""
    _check_sounddevice()
    import sounddevice as sd

    try:
        import config
        sample_rate = config.AUDIO_SAMPLE_RATE
        vad_threshold = config.VAD_THRESHOLD
        deaf_window_secs = config.POST_SPEECH_LISTEN_DELAY_SECS
        aec_tail_secs = config.POST_PLAYBACK_SUPPRESSION_SECS
    except Exception as exc:
        _die(f"Could not load config: {exc}")

    device_idx = _get_device_index()
    if device_idx is None:
        _list_input_devices()
        _die("AUDIO_DEVICE_NAME/AUDIO_DEVICE_INDEX not set or not found in .env.")

    vad_model, vad_ok = _load_silero()

    # Import AEC module to query live suppression state
    try:
        from audio import echo_cancel as _aec
        aec_available = True
    except Exception:
        aec_available = False

    print(f"{ts()} VAD test")
    print(f"{ts()} Device: {device_idx}  Sample rate: {sample_rate} Hz")
    print(f"{ts()} VAD threshold:   {vad_threshold}  (Silero probability cutoff)")
    print(f"{ts()} Deaf window:     {deaf_window_secs}s (POST_SPEECH_LISTEN_DELAY_SECS — blocks speech onset after TTS)")
    print(f"{ts()} AEC tail:        {aec_tail_secs}s (POST_PLAYBACK_SUPPRESSION_SECS — mic attenuation after playback ends)")
    print(f"{ts()} Silero VAD:      {'loaded' if vad_ok else 'UNAVAILABLE — will use RMS fallback'}")
    print(f"{ts()} AEC module:      {'available' if aec_available else 'unavailable'}")
    print(f"{ts()} Press Ctrl+C to stop.")
    print()

    blocksize = int(sample_rate * 0.032)
    chunk_queue: list = []
    q_lock = threading.Lock()
    stop = threading.Event()

    def callback(indata, frames, t, status):
        with q_lock:
            chunk_queue.append(indata[:, 0].copy())

    def vad_loop():
        while not stop.is_set():
            with q_lock:
                chunk = chunk_queue.pop(0) if chunk_queue else None
            if chunk is None:
                time.sleep(0.005)
                continue

            if vad_ok:
                prob = _silero_prob(vad_model, chunk, sample_rate)
                is_sp = prob >= vad_threshold
            else:
                rms = float(np.sqrt(np.mean(chunk ** 2)))
                prob = min(1.0, rms / 0.05)
                is_sp = rms > 0.01

            label = "SPEECH " if is_sp else "SILENCE"

            # AEC suppression state
            if aec_available:
                suppressed = _aec.is_suppressed()
                aec_str = f"[aec: {'ACTIVE  ' if suppressed else 'inactive'}]"
            else:
                aec_str = "[aec: unavailable]"

            # Deaf window: inactive in standalone test (no TTS running).
            # In the full system this is set for POST_SPEECH_LISTEN_DELAY_SECS after each TTS.
            deaf_str = f"[deaf_window: inactive — no TTS in this test]"

            sys.stdout.write(
                f"\r{ts()} {label}  conf={prob:.2f}  {deaf_str}  {aec_str}   "
            )
            sys.stdout.flush()

    try:
        with sd.InputStream(
            device=device_idx,
            samplerate=sample_rate,
            channels=1,
            dtype="float32",
            blocksize=blocksize,
            callback=callback,
        ):
            t = threading.Thread(target=vad_loop, daemon=True)
            t.start()
            while True:
                time.sleep(0.1)
    except KeyboardInterrupt:
        stop.set()
        print(f"\n{ts()} Stopped.")
    except sd.PortAudioError as exc:
        stop.set()
        _die(f"PortAudio error on device {device_idx}: {exc}")


# ─────────────────────────────────────────────────────────────────────────────
# --test aec
# ─────────────────────────────────────────────────────────────────────────────

def test_aec() -> None:
    """Echo cancellation: play a cached TTS file and watch mic suppression."""
    _check_sounddevice()

    try:
        import sounddevice as sd
        import soundfile as sf
    except ImportError as exc:
        _die(f"Missing dependency: {exc}\nRun: pip install soundfile sounddevice")

    try:
        import config
        sample_rate = config.AUDIO_SAMPLE_RATE
        cache_dir = Path(config.TTS_CACHE_DIR)
        suppression_factor = config.AEC_SUPPRESSION_FACTOR
        tail_secs = config.POST_PLAYBACK_SUPPRESSION_SECS
    except Exception as exc:
        _die(f"Could not load config: {exc}")

    device_idx = _get_device_index()
    if device_idx is None:
        _list_input_devices()
        _die("AUDIO_DEVICE_NAME/AUDIO_DEVICE_INDEX not set or not found in .env.")

    # Pick first cached TTS file
    mp3_files = sorted(cache_dir.glob("*.mp3")) if cache_dir.exists() else []
    if not mp3_files:
        _die(
            f"No TTS cache files found in {cache_dir}\n"
            f"  → Run the main system briefly to generate cached audio, or copy any MP3 into {cache_dir}"
        )

    cache_file = mp3_files[0]

    try:
        audio_data, audio_sr = sf.read(str(cache_file), dtype="float32", always_2d=False)
        if audio_data.ndim > 1:
            audio_data = audio_data.mean(axis=1)
    except Exception as exc:
        _die(f"Could not decode {cache_file.name}: {exc}\nRun: pip install soundfile")

    duration = len(audio_data) / audio_sr

    # Import echo_cancel (its internal stream.flush() is a no-op here since stream isn't started)
    try:
        from audio import echo_cancel
    except Exception as exc:
        _die(f"Could not import audio.echo_cancel: {exc}")

    print(f"{ts()} AEC test")
    print(f"{ts()} Cache file:         {cache_file.name}")
    print(f"{ts()} Audio duration:     {duration:.2f}s  (sample rate: {audio_sr} Hz)")
    print(f"{ts()} Suppression factor: {suppression_factor} ({suppression_factor*100:.0f}% of mic input during playback)")
    print(f"{ts()} Post-play tail:     {tail_secs}s")
    print()

    # Mic monitoring
    rms_history: list = []
    rms_lock = threading.Lock()
    mic_stop = threading.Event()

    def mic_callback(indata, frames, t, status):
        rms = float(np.sqrt(np.mean(indata[:, 0] ** 2)))
        with rms_lock:
            rms_history.append((time.monotonic(), rms))

    def mic_printer():
        while not mic_stop.is_set():
            time.sleep(0.05)
            with rms_lock:
                entry = rms_history[-1] if rms_history else None
            if entry is None:
                continue
            _, rms = entry
            suppressed = echo_cancel.is_suppressed()
            state_label = "SUPPRESSED" if suppressed else "normal    "
            bar = _bar(rms / 0.08, 25)
            sys.stdout.write(f"\r{ts()} mic RMS: {bar} {rms:.5f}  [{state_label}]   ")
            sys.stdout.flush()

    mic_stream = sd.InputStream(
        device=device_idx,
        samplerate=sample_rate,
        channels=1,
        dtype="float32",
        blocksize=512,
        callback=mic_callback,
    )

    try:
        mic_stream.start()
        printer_thread = threading.Thread(target=mic_printer, daemon=True)
        printer_thread.start()

        # Measure baseline RMS for ~0.5s
        time.sleep(0.5)
        with rms_lock:
            baseline_vals = [r for _, r in rms_history]
        baseline = sum(baseline_vals) / len(baseline_vals) if baseline_vals else 0.0
        print(f"\n{ts()} Baseline mic RMS: {baseline:.5f}")
        print()

        # Suppression START
        t_start = time.monotonic()
        print(f"{ts()} SUPPRESSION START  — echo_cancel.set_playing(True)")
        echo_cancel.set_playing(True)

        # Play the cached TTS file
        sd.play(audio_data, audio_sr)
        sd.wait()

        t_end = time.monotonic()
        print(f"\n{ts()} Playback finished  — duration {t_end - t_start:.2f}s")

        # Suppression END  → tail activates
        echo_cancel.set_playing(False)
        t_stop_called = time.monotonic()
        print(f"{ts()} SUPPRESSION END    — echo_cancel.set_playing(False), {tail_secs}s tail active")
        print(f"{ts()} BUFFER FLUSH       — mic buffer cleared (via echo_cancel internals)")

        # Wait for tail to expire
        time.sleep(tail_secs + 0.15)
        print(f"\n{ts()} TAIL END           — {tail_secs}s elapsed since playback stopped")

        if echo_cancel.is_suppressed():
            print(f"{ts()} WARNING: AEC still showing suppressed after tail expired — check POST_PLAYBACK_SUPPRESSION_SECS")
        else:
            print(f"{ts()} AEC confirmed inactive")

        # Post-suppression RMS
        time.sleep(0.3)
        with rms_lock:
            post_vals = [r for _, r in rms_history[-20:]]
        post_rms = sum(v for _, v in post_vals) / len(post_vals) if post_vals else 0.0
        print(f"{ts()} Post-suppression RMS: {post_rms:.5f}  (baseline was {baseline:.5f})")
        print()
        print(f"{ts()} AEC test complete.")

    except KeyboardInterrupt:
        echo_cancel.set_playing(False)
        print(f"\n{ts()} Interrupted.")
    finally:
        mic_stop.set()
        try:
            mic_stream.stop()
            mic_stream.close()
        except Exception:
            pass


# ─────────────────────────────────────────────────────────────────────────────
# --test transcription
# ─────────────────────────────────────────────────────────────────────────────

def test_transcription() -> None:
    """Record audio then run it through the full transcription pipeline."""
    _check_sounddevice()
    import sounddevice as sd

    try:
        import config
        sample_rate = config.AUDIO_SAMPLE_RATE
        vad_threshold = config.VAD_THRESHOLD
        silence_timeout = config.SILENCE_TIMEOUT_SECS
        corrections = config.WHISPER_CORRECTIONS
        hallucination_blocklist = config.HALLUCINATION_BLOCKLIST
        whisper_initial_prompt = config.WHISPER_INITIAL_PROMPT
        whisper_local_dir = (_ROOT / config.WHISPER_MODEL_DIR).resolve()
    except Exception as exc:
        _die(f"Could not load config: {exc}")

    device_idx = _get_device_index()
    if device_idx is None:
        _list_input_devices()
        _die("AUDIO_DEVICE_NAME/AUDIO_DEVICE_INDEX not set or not found in .env.")

    # Check backends
    try:
        import mlx_whisper
        mlx_ok = True
    except ImportError:
        mlx_ok = False

    local_model_ready = (whisper_local_dir / "config.json").exists()

    openai_ok = False
    try:
        import apikeys as _ak
        key = getattr(_ak, "OPENAI_API_KEY", "")
        openai_ok = bool(key) and not key.lower().startswith("your") and not key.endswith("...")
    except Exception:
        pass

    print(f"{ts()} Transcription test")
    print(f"{ts()} mlx_whisper installed:  {'YES' if mlx_ok else 'NO  → pip install mlx-whisper'}")
    print(f"{ts()} Local model dir:        {whisper_local_dir}")
    print(f"{ts()} Local model ready:      {'YES' if local_model_ready else 'NO  → run: python setup_assets.py'}")
    print(f"{ts()} OpenAI fallback key:    {'SET' if openai_ok else 'MISSING — set OPENAI_API_KEY in apikeys.py'}")
    print(f"{ts()} Corrections map:        {len(corrections)} entries")
    print(f"{ts()} Hallucination blocklist:{len(hallucination_blocklist)} entries")
    print(f"{ts()} Silence timeout:        {silence_timeout}s")
    print()

    if not mlx_ok and not openai_ok:
        _die(
            "No transcription backend available.\n"
            "  → Install mlx-whisper  OR  set OPENAI_API_KEY in apikeys.py"
        )
    if mlx_ok and not local_model_ready and not openai_ok:
        _die("mlx_whisper installed but local model missing, and no OpenAI fallback key.\nRun: python setup_assets.py")

    vad_model, vad_ok = _load_silero()

    print(f"{ts()} Speak now. Recording stops after {silence_timeout}s silence or Ctrl+C.")
    print(f"{ts()} Up to 10s total silence before auto-stop.")
    print()

    blocksize = 512
    chunk_queue: list = []
    q_lock = threading.Lock()

    def callback(indata, frames, t, status):
        with q_lock:
            chunk_queue.append(indata[:, 0].copy())

    def _run_transcription_pipeline(audio_arr: np.ndarray, seg_num: int) -> None:
        duration = len(audio_arr) / sample_rate
        print(f"\n{ts()} ── Segment {seg_num} ({duration:.2f}s) ─────────────────────────")

        # Which backend will be used?
        if mlx_ok and local_model_ready:
            expected_backend = "mlx_whisper"
        elif openai_ok:
            expected_backend = "openai_whisper (fallback)"
        else:
            expected_backend = "none"
        print(f"{ts()} Expected backend:      {expected_backend}")

        # Step 1: raw Whisper output
        raw = ""
        actual_backend = "none"

        if mlx_ok and local_model_ready:
            try:
                result = mlx_whisper.transcribe(
                    audio_arr,
                    path_or_hf_repo=str(whisper_local_dir),
                    initial_prompt=whisper_initial_prompt,
                )
                raw = result.get("text", "").strip()
                actual_backend = "mlx_whisper"
            except Exception as exc:
                print(f"{ts()} mlx_whisper error: {exc} — trying OpenAI fallback")

        if not raw and openai_ok:
            try:
                import io, wave, apikeys
                from openai import OpenAI
                buf = io.BytesIO()
                with wave.open(buf, "wb") as wf:
                    wf.setnchannels(1)
                    wf.setsampwidth(2)
                    wf.setframerate(sample_rate)
                    pcm = (audio_arr * 32767).clip(-32768, 32767).astype(np.int16)
                    wf.writeframes(pcm.tobytes())
                buf.seek(0)
                buf.name = "audio.wav"
                client = OpenAI(api_key=apikeys.OPENAI_API_KEY)
                resp = client.audio.transcriptions.create(
                    model=config.WHISPER_FALLBACK_MODEL,
                    file=buf,
                    prompt=whisper_initial_prompt,
                )
                raw = resp.text.strip()
                actual_backend = "openai_whisper"
            except Exception as exc:
                print(f"{ts()} OpenAI Whisper error: {exc}")

        print(f"{ts()} Actual backend:        {actual_backend}")
        print(f"{ts()} Raw Whisper output:    {raw!r}")

        if not raw:
            print(f"{ts()} (empty output — nothing to process)")
            return

        # Step 2: hallucination filter
        import re
        from collections import Counter
        lower = raw.lower().strip()
        normalized = re.sub(r"\s+", " ", re.sub(r"[^a-z0-9'\s]", " ", lower)).strip()

        blocklist_match = None
        for phrase in hallucination_blocklist:
            phrase_norm = re.sub(r"\s+", " ", re.sub(r"[^a-z0-9'\s]", " ", phrase.lower())).strip()
            if normalized == phrase_norm:
                blocklist_match = phrase
                break

        words = re.findall(r"[a-z0-9']+", normalized)
        repeat_word = None
        if words:
            top = Counter(words).most_common(1)[0]
            if top[1] > 5:
                repeat_word = top[0]

        is_halluc = blocklist_match is not None or repeat_word is not None
        if is_halluc:
            reason = f"blocklist match: {blocklist_match!r}" if blocklist_match else f"repeated word '{repeat_word}' >5×"
            print(f"{ts()} Hallucination filter:  FILTERED ({reason})")
            print(f"{ts()} Final output:          '' (filtered)")
            return
        else:
            print(f"{ts()} Hallucination filter:  PASS")

        # Step 3: corrections map
        corrected = raw
        applied = []
        for pattern, replacement in corrections.items():
            if re.search(re.escape(pattern), corrected, re.IGNORECASE):
                corrected = re.sub(re.escape(pattern), replacement, corrected, flags=re.IGNORECASE)
                applied.append(f"{pattern!r} → {replacement!r}")

        if applied:
            print(f"{ts()} Corrections applied:   {', '.join(applied)}")
        else:
            print(f"{ts()} Corrections applied:   none")

        print(f"{ts()} Final output:          {corrected!r}")
        print(f"{ts()} OpenAI fallback used:  {'YES' if actual_backend == 'openai_whisper' else 'NO'}")

    # Recording loop
    in_speech = False
    speech_chunks: list = []
    last_speech_at = time.monotonic()
    total_silence_secs = 0.0
    MAX_IDLE_SILENCE = 10.0
    segment_count = 0
    stop = threading.Event()

    try:
        with sd.InputStream(
            device=device_idx,
            samplerate=sample_rate,
            channels=1,
            dtype="float32",
            blocksize=blocksize,
            callback=callback,
        ):
            print(f"{ts()} Mic open. Listening...")
            while not stop.is_set():
                time.sleep(0.016)
                with q_lock:
                    chunks = list(chunk_queue)
                    chunk_queue.clear()

                for chunk in chunks:
                    if vad_ok:
                        prob = _silero_prob(vad_model, chunk, sample_rate)
                        is_sp = prob >= vad_threshold
                    else:
                        is_sp = float(np.sqrt(np.mean(chunk ** 2))) > 0.01

                    if is_sp:
                        if not in_speech:
                            print(f"{ts()} Speech detected — accumulating", end="", flush=True)
                        in_speech = True
                        speech_chunks.append(chunk)
                        last_speech_at = time.monotonic()
                        total_silence_secs = 0.0
                        sys.stdout.write(".")
                        sys.stdout.flush()
                    else:
                        if in_speech:
                            elapsed = time.monotonic() - last_speech_at
                            if elapsed >= silence_timeout:
                                in_speech = False
                                print(f"\n{ts()} Silence {elapsed:.1f}s — processing...")
                                audio_arr = np.concatenate(speech_chunks)
                                speech_chunks.clear()
                                segment_count += 1
                                _run_transcription_pipeline(audio_arr, segment_count)
                                last_speech_at = time.monotonic()
                        else:
                            total_silence_secs += len(chunk) / sample_rate
                            if total_silence_secs >= MAX_IDLE_SILENCE:
                                print(f"{ts()} {MAX_IDLE_SILENCE:.0f}s of idle silence — stopping.")
                                stop.set()

    except KeyboardInterrupt:
        print(f"\n{ts()} Ctrl+C — stopping.")
        if speech_chunks:
            audio_arr = np.concatenate(speech_chunks)
            if len(audio_arr) >= sample_rate * 0.3:
                segment_count += 1
                print(f"{ts()} Transcribing buffered audio...")
                _run_transcription_pipeline(audio_arr, segment_count)

    if segment_count == 0:
        print(f"{ts()} No speech segments captured.")


# ─────────────────────────────────────────────────────────────────────────────
# --test pipeline
# ─────────────────────────────────────────────────────────────────────────────

def test_pipeline() -> None:
    """Full end-to-end single interaction cycle."""
    _check_sounddevice()
    import sounddevice as sd

    # config_loader validates API keys — give a clear error if they're missing
    try:
        import config
        from utils.config_loader import AUDIO_DEVICE_INDEX  # noqa: F401 (validates/resolves audio config)
    except RuntimeError as exc:
        _die(str(exc))
    except Exception as exc:
        _die(f"Config error: {exc}")

    sample_rate = config.AUDIO_SAMPLE_RATE
    vad_threshold = config.VAD_THRESHOLD
    silence_timeout = config.SILENCE_TIMEOUT_SECS
    deaf_window = config.POST_SPEECH_LISTEN_DELAY_SECS
    skip_wake = config.IDLE_LISTEN_WITHOUT_WAKE_WORD

    device_idx = _get_device_index()
    if device_idx is None:
        _list_input_devices()
        _die("AUDIO_DEVICE_NAME/AUDIO_DEVICE_INDEX not set or not found in .env.")

    print(f"{ts()} Pipeline test — full single interaction cycle")
    print(f"{ts()} Device: {device_idx}  Sample rate: {sample_rate} Hz")
    print(f"{ts()} IDLE_LISTEN_WITHOUT_WAKE_WORD = {skip_wake}")
    print(f"{ts()} Silence timeout: {silence_timeout}s  Deaf window: {deaf_window}s")
    print()

    # Load VAD
    vad_model, vad_ok = _load_silero()
    print(f"{ts()} Silero VAD: {'loaded' if vad_ok else 'unavailable — RMS fallback'}")

    # Import audio modules
    from audio import stream as stream_module
    from audio import echo_cancel
    import random

    # Import TTS — wraps gracefully if hardware (leds_head) fails
    tts_module = None
    try:
        from audio import tts as tts_module
        print(f"{ts()} TTS module: loaded")
    except Exception as exc:
        print(f"{ts()} WARNING: TTS unavailable ({exc}) — speech step will be skipped")

    # Start audio stream
    print(f"\n{ts()} Starting audio stream...")
    stream_module.start()
    time.sleep(0.3)
    print(f"{ts()} Audio stream running.")
    print()

    blocksize = 512
    chunk_queue: list = []
    q_lock = threading.Lock()
    stop_listen = threading.Event()
    wake_fired = threading.Event()
    wake_name: list = [None]

    def mic_callback(indata, frames, t, status):
        with q_lock:
            chunk_queue.append(indata[:, 0].copy())

    try:
        # ── Step 1: Wake word (or passive listen) ──────────────────────────────
        if not skip_wake:
            print(f"{ts()} Waiting for wake word (say 'Hey Rex', 'Hey DJ Rex', etc.)...")
            try:
                from audio import wake_word as ww
                from state import State
                import state as state_module
                state_module.set_state(State.IDLE)

                def on_wake(name: str) -> None:
                    wake_name[0] = name
                    wake_fired.set()

                ww.start(on_wake)
                if not wake_fired.wait(timeout=60.0):
                    ww.stop()
                    _die("No wake word in 60s. Check models in assets/models/wake_word/")
                ww.stop()
                print(f"{ts()} Wake word detected: {wake_name[0]}")
            except Exception as exc:
                _die(f"Wake word module failed: {exc}\nCheck that model files exist in assets/models/wake_word/")

            # Acknowledgment via TTS
            ack = random.choice(config.WAKE_ACKNOWLEDGMENTS)
            print(f"{ts()} Speaking acknowledgment: {ack!r}")
            if tts_module:
                tts_module.speak(ack)
            else:
                print(f"{ts()} (TTS unavailable — skipping speech)")

            # Deaf window: discard any speech that arrived during/after TTS
            print(f"{ts()} Deaf window active: {deaf_window}s")
            resume_at = time.monotonic() + deaf_window
            while time.monotonic() < resume_at:
                time.sleep(0.05)
            stream_module.flush()
            print(f"{ts()} Deaf window expired — now listening for speech")
        else:
            print(f"{ts()} Passive listen (no wake word required)")

        # ── Step 2: VAD-gated speech accumulation ──────────────────────────────
        print(f"{ts()} Listening for speech... (speak now)")
        speech_chunks: list = []
        in_speech = False
        last_speech_at = time.monotonic()

        with sd.InputStream(
            device=device_idx,
            samplerate=sample_rate,
            channels=1,
            dtype="float32",
            blocksize=blocksize,
            callback=mic_callback,
        ):
            while not stop_listen.is_set():
                time.sleep(0.016)
                with q_lock:
                    chunks = list(chunk_queue)
                    chunk_queue.clear()

                for chunk in chunks:
                    # AEC: skip chunks while Rex is playing
                    if echo_cancel.is_suppressed():
                        continue

                    if vad_ok:
                        prob = _silero_prob(vad_model, chunk, sample_rate)
                        is_sp = prob >= vad_threshold
                    else:
                        is_sp = float(np.sqrt(np.mean(chunk ** 2))) > 0.01

                    if is_sp:
                        if not in_speech:
                            print(f"{ts()} Speech onset — accumulating", end="", flush=True)
                        in_speech = True
                        speech_chunks.append(chunk)
                        last_speech_at = time.monotonic()
                        sys.stdout.write(".")
                        sys.stdout.flush()
                    else:
                        if in_speech and (time.monotonic() - last_speech_at) >= silence_timeout:
                            in_speech = False
                            print(f"\n{ts()} Silence — speech segment complete")
                            stop_listen.set()

        if not speech_chunks:
            print(f"{ts()} No speech captured — exiting.")
            return

        audio_arr = np.concatenate(speech_chunks)
        duration = len(audio_arr) / sample_rate
        print(f"{ts()} Captured {duration:.2f}s of speech")

        # ── Step 3: Transcription ───────────────────────────────────────────────
        print(f"\n{ts()} ── Transcribing ─────────────────────────────────────────────")
        from audio.transcription import transcribe
        transcript = transcribe(audio_arr)
        print(f"{ts()} Transcript: {transcript!r}")

        if not transcript:
            print(f"{ts()} Empty transcript (silence or hallucination filtered) — exiting.")
            return

        # ── Step 4: Command parser ──────────────────────────────────────────────
        print(f"\n{ts()} ── Command parser ───────────────────────────────────────────")
        from intelligence.command_parser import parse
        match = parse(transcript)

        response_text = ""
        if match:
            print(f"{ts()} Match: {match.command_key}  ({match.match_type})")
            if match.args:
                print(f"{ts()} Args:  {match.args}")
            response_text = f"Roger that. Handling {match.command_key}."
            print(f"{ts()} Local command response: {response_text!r}")
        else:
            print(f"{ts()} No command match — sending to LLM")
            print(f"\n{ts()} ── LLM call ─────────────────────────────────────────────────")
            response_text = _pipeline_llm_call(transcript)
            print(f"{ts()} LLM response: {response_text!r}")

        # ── Step 5: TTS with AEC ────────────────────────────────────────────────
        if response_text:
            print(f"\n{ts()} ── TTS output ───────────────────────────────────────────────")
            if tts_module:
                print(f"{ts()} Speaking (AEC suppression will activate)...")
                tts_module.speak(response_text)
                print(f"{ts()} TTS complete — AEC suppression ended")
            else:
                print(f"{ts()} TTS unavailable — response would be: {response_text!r}")

        print(f"\n{ts()} Pipeline cycle complete — exiting.")

    except KeyboardInterrupt:
        print(f"\n{ts()} Interrupted.")
    finally:
        stream_module.stop()


def _pipeline_llm_call(text: str) -> str:
    """Minimal direct OpenAI call for the pipeline test — bypasses the full llm.py machinery."""
    try:
        import apikeys
        from openai import OpenAI
        client = OpenAI(api_key=apikeys.OPENAI_API_KEY)
        response = client.chat.completions.create(
            model=config.LLM_MODEL,
            messages=[
                {
                    "role": "system",
                    "content": (
                        "You are DJ-R3X, a wisecracking animatronic droid DJ at Oga's Cantina "
                        "in Star Wars: Galaxy's Edge. Keep your response to 1-2 sentences."
                    ),
                },
                {"role": "user", "content": text},
            ],
            max_tokens=80,
        )
        return response.choices[0].message.content.strip()
    except Exception as exc:
        print(f"{ts()} LLM call failed: {exc}")
        return ""


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(
        description="DJ-R3X v2 audio pipeline diagnostics",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
test modes:
  mic            Real-time mic level meter — verify configured mic and signal
  vad            Real-time Silero VAD decisions with confidence scores
  aec            Play a cached TTS file and confirm mic suppression activates
  transcription  Record until Ctrl+C or 10s silence, run full transcription pipeline
  pipeline       Full single interaction cycle: wake → listen → transcribe → respond → speak
""",
    )
    parser.add_argument(
        "--test",
        required=True,
        choices=["mic", "vad", "aec", "transcription", "pipeline"],
    )
    args = parser.parse_args()

    {
        "mic": test_mic,
        "vad": test_vad,
        "aec": test_aec,
        "transcription": test_transcription,
        "pipeline": test_pipeline,
    }[args.test]()


if __name__ == "__main__":
    main()
