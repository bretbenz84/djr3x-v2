#!/usr/bin/env python3
"""
rex_supervisor.py — the always-on "wake up Rex" launcher.

This is a deliberately tiny, dependency-light process meant to run for the whole
login session (via a macOS LaunchAgent). It does ONE thing: listen for the single
wake word "wake up rex" (wakeuprex.onnx) and, when it hears it, launch the full
DJ-R3X controller (main.py in the project venv).

It is intentionally simple: just the openWakeWord ONNX model. No VAD, no Whisper,
no transcription — the robot is OFF while the supervisor listens, so the only
thing that ever needs to happen is detecting one wake word and launching main.py.

Why a separate process instead of just running main.py at login:
  - The robot stays "off" (no servos waking, no camera, no LLM) until you summon
    it by voice, but the Mac is always ready to listen.
  - "shut down" / "shut down rex" cleanly exits main.py and hands control back
    here, so you can power the droid down without killing this listener.

The coordination that prevents a DOUBLE launch (the tricky case):
  main.py holds a single-instance flock for its entire lifetime — including while
  it is merely ASLEEP (the "go to sleep" state, which only wakes on its own
  internal "wake up rex" detector). This supervisor checks that lock and stays
  DORMANT whenever a controller is alive. So:
    - main.py awake  → lock held → supervisor dormant (main.py owns the mic)
    - main.py asleep → lock held → supervisor dormant (main.py's own wake word
                                    handles waking; we must NOT spawn a 2nd one)
    - no main.py     → lock free → supervisor listens for "wake up rex"
  The flock auto-frees if main.py crashes, so the supervisor resumes on its own.

Only one process listens to the mic at a time, so there is no contention.

Run directly for debugging:
    venv/bin/python rex_supervisor.py
"""

from __future__ import annotations

import logging
import os
import signal
import subprocess
import sys
import threading
import time
from pathlib import Path
from typing import Optional

_PROJECT_ROOT = Path(__file__).resolve().parent
_VENV_PYTHON = _PROJECT_ROOT / "venv" / "bin" / "python"
_WAKE_MODEL = _PROJECT_ROOT / "assets" / "models" / "wake_word" / "wakeuprex.onnx"
# Short chime played the instant a wake word is accepted, so there's immediate
# feedback before the (slower) full controller finishes booting.
_CHIME_FILE = _PROJECT_ROOT / "assets" / "audio" / "startup" / "startup_chime.mp3"

# 80 ms at 16 kHz — openWakeWord's preferred sequential frame size.
_SAMPLE_RATE = 16000
_CHUNK_SAMPLES = 1280
_CHUNK_SECS = _CHUNK_SAMPLES / _SAMPLE_RATE

_DEBUG = os.environ.get("REX_SUPERVISOR_DEBUG", "").strip() in ("1", "true", "True")
# Play the startup chime on wake (instant feedback before the robot boots).
# Set REX_SUPERVISOR_CHIME=0 to disable.
_CHIME_ENABLED = os.environ.get("REX_SUPERVISOR_CHIME", "1").strip() not in ("0", "false", "False")

# Make utils.single_instance importable without importing the heavy project config.
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

logging.basicConfig(
    level=logging.DEBUG if _DEBUG else logging.INFO,
    format="%(asctime)s | rex_supervisor | %(levelname)s | %(message)s",
)
log = logging.getLogger("rex_supervisor")

_stop = threading.Event()


# ── Minimal .env reading (no project config import) ────────────────────────────

def _read_env_file() -> dict[str, str]:
    """Parse KEY=VALUE lines from .env without importing the project config.

    The supervisor must start even when apikeys.py / full config would fail, so
    it reads only what it needs (the mic device) straight from .env.
    """
    env: dict[str, str] = {}
    path = _PROJECT_ROOT / ".env"
    try:
        for raw in path.read_text().splitlines():
            line = raw.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            key, _, value = line.partition("=")
            value = value.strip()
            # Strip a matching pair of surrounding quotes — .env values like
            # AUDIO_DEVICE_NAME="MacBook Pro Microphone" must resolve to the bare
            # device name, not include the literal quotes (which never match).
            if len(value) >= 2 and value[0] == value[-1] and value[0] in ("'", '"'):
                value = value[1:-1]
            env[key.strip()] = value
    except OSError:
        pass
    return env


def _list_input_devices():
    """Return [(index, name), ...] for devices with at least one input channel."""
    import sounddevice as sd
    out = []
    try:
        for idx, dev in enumerate(sd.query_devices()):
            try:
                if int(dev.get("max_input_channels", 0)) > 0:
                    out.append((idx, str(dev.get("name") or "").strip()))
            except Exception:
                continue
    except Exception as exc:
        log.warning("Could not query audio devices: %s", exc)
    return out


def _device_max_input_channels(device) -> int:
    """Max input channels the (resolved) device exposes; 0/unknown → 0."""
    try:
        import sounddevice as sd
        info = sd.query_devices(device if device is not None else None, kind="input")
        return int(info.get("max_input_channels", 0) or 0)
    except Exception:
        return 0


def _resolve_input_device(env: dict[str, str]):
    """Resolve a sounddevice input device from .env, else the system default.

    Mirrors the main app's tolerant matching (utils.config_loader): exact name
    match first (case-insensitive), then a unique substring match, then the
    AUDIO_DEVICE_INDEX fallback, then the system default.
    """
    name = (os.environ.get("AUDIO_DEVICE_NAME") or env.get("AUDIO_DEVICE_NAME") or "").strip()
    index_raw = (os.environ.get("AUDIO_DEVICE_INDEX") or env.get("AUDIO_DEVICE_INDEX") or "").strip()

    inputs = _list_input_devices()

    if name:
        wanted = name.lower()
        exact = [(idx, nm) for idx, nm in inputs if nm.lower() == wanted]
        if exact:
            return exact[0][0]
        contains = [(idx, nm) for idx, nm in inputs if wanted in nm.lower()]
        if len(contains) == 1:
            return contains[0][0]
        if len(contains) > 1:
            opts = ", ".join(f"{idx}:{nm}" for idx, nm in contains)
            log.warning("AUDIO_DEVICE_NAME=%r matched multiple inputs (%s) — be more specific.", name, opts)
        else:
            avail = ", ".join(f"{idx}:{nm}" for idx, nm in inputs) or "no input devices"
            log.warning("AUDIO_DEVICE_NAME=%r did not match any input. Available: %s", name, avail)

    if index_raw:
        try:
            return int(index_raw)
        except ValueError:
            log.warning("AUDIO_DEVICE_INDEX=%r is not an integer; using default.", index_raw)
    return None  # sounddevice picks the default input


def _device_label(device) -> str:
    """Human-readable name for a resolved device index (or 'default')."""
    if device is None:
        try:
            import sounddevice as sd
            di = sd.query_devices(kind="input")
            return f"default ({str(di.get('name','?'))})"
        except Exception:
            return "default"
    try:
        import sounddevice as sd
        return f"{device}:{str(sd.query_devices(device).get('name','?'))}"
    except Exception:
        return str(device)


# ── Audio scaling for openWakeWord ─────────────────────────────────────────────

def _to_oww_input(mono):
    """Scale a mono float32 [-1,1] array to int16-range PCM for openWakeWord.

    THIS IS LOAD-BEARING: openWakeWord's melspectrogram front-end is trained on
    16-bit PCM (range ±32767). Feeding the raw float [-1,1] that sounddevice
    returns makes the model see near-silence, so every score pins at ~0.001 and
    the wake word NEVER fires — the real reason "I said wake up Rex and nothing
    happened." Scaling to int16 makes a clear "wake up rex" score ~0.99.
    """
    import numpy as np
    return (np.clip(mono, -1.0, 1.0) * 32767.0).astype(np.int16)


# ── Controller liveness ────────────────────────────────────────────────────────

def _controller_running(child: Optional[subprocess.Popen]) -> bool:
    """True if a DJ-R3X controller is alive (our child, or any lock holder)."""
    if child is not None and child.poll() is None:
        return True
    try:
        from utils import single_instance
        return single_instance.is_held_by_other()
    except Exception as exc:
        log.debug("single_instance check failed: %s", exc)
        return False


def _play_chime() -> None:
    """Play the startup chime as immediate wake feedback. Fire-and-forget, in a
    separate process so it never blocks the wake loop or touches the mic.

    Prefers macOS `afplay` (built in, decodes MP3, no Python deps). Falls back to
    soundfile + sounddevice if afplay is unavailable.
    """
    if not _CHIME_ENABLED:
        return
    if not _CHIME_FILE.exists():
        log.warning("Startup chime missing: %s", _CHIME_FILE)
        return
    import shutil
    afplay = shutil.which("afplay")
    if afplay:
        try:
            subprocess.Popen(
                [afplay, str(_CHIME_FILE)],
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
            )
            return
        except Exception as exc:
            log.debug("afplay chime failed (%s) — trying soundfile.", exc)

    def _play_blocking():
        try:
            import soundfile as sf
            import sounddevice as sd
            audio, sr = sf.read(str(_CHIME_FILE), dtype="float32", always_2d=False)
            sd.play(audio, sr)
            sd.wait()
        except Exception as exc:
            log.debug("soundfile chime playback failed: %s", exc)

    threading.Thread(target=_play_blocking, daemon=True, name="rex-chime").start()


def _launch_controller() -> Optional[subprocess.Popen]:
    """Start main.py in the project venv as a detached child."""
    if not _VENV_PYTHON.exists():
        log.error("venv python not found at %s — cannot launch controller.", _VENV_PYTHON)
        return None
    log.info("Wake word heard — launching DJ-R3X controller.")
    try:
        return subprocess.Popen(
            [str(_VENV_PYTHON), str(_PROJECT_ROOT / "main.py")],
            cwd=str(_PROJECT_ROOT),
        )
    except Exception as exc:
        log.error("Failed to launch controller: %s", exc)
        return None


# ── Wake-word model ────────────────────────────────────────────────────────────

def _feature_model_kwargs() -> dict:
    """Point openWakeWord at the repo's bundled feature models when the pip
    package is missing its own (same self-heal as audio/wake_word.py)."""
    try:
        import openwakeword as oww
        melspec_default = Path(
            oww.FEATURE_MODELS["melspectrogram"]["model_path"]
        ).with_suffix(".onnx")
        embedding_default = Path(
            oww.FEATURE_MODELS["embedding"]["model_path"]
        ).with_suffix(".onnx")
        if melspec_default.exists() and embedding_default.exists():
            return {}
    except Exception:
        pass

    res = _PROJECT_ROOT / "assets" / "models" / "wake_word" / "_openwakeword_resources"
    melspec = res / "melspectrogram.onnx"
    embedding = res / "embedding_model.onnx"
    if melspec.exists() and embedding.exists():
        return {
            "melspec_model_path": str(melspec),
            "embedding_model_path": str(embedding),
        }
    return {}


def _load_model():
    """Load ONLY the wakeuprex model via openWakeWord."""
    try:
        from openwakeword.model import Model
    except ImportError:
        log.error("openwakeword not installed in venv — supervisor cannot listen.")
        return None
    if not _WAKE_MODEL.exists():
        log.error("Wake model missing: %s", _WAKE_MODEL)
        return None
    try:
        return Model(
            wakeword_models=[str(_WAKE_MODEL)],
            inference_framework="onnx",
            **_feature_model_kwargs(),
        )
    except Exception as exc:
        log.error("Failed to initialise wakeuprex model: %s", exc)
        return None


def _wake_threshold() -> float:
    # 0.7 (not 0.5) so background TV/ambient — which tops out around 0.12 in
    # practice — can't graze the bar. A clean "wake up rex" scores ~0.99, so
    # there's wide margin. Lower it only if a real phrase won't trigger.
    try:
        return float(os.environ.get("REX_SUPERVISOR_WAKE_THRESHOLD", "0.7"))
    except ValueError:
        return 0.7


def _wake_consecutive() -> int:
    """How many CONSECUTIVE 80 ms frames must clear the threshold before firing.

    A real "wake up rex" holds the model near 1.0 for ~10 frames in a row; a TV
    phonetic near-miss is a 1-2 frame spike. Requiring a short sustained run
    (default 3 ≈ 240 ms) kills those spikes without risking real wakes.
    """
    try:
        return max(1, int(os.environ.get("REX_SUPERVISOR_WAKE_CONSECUTIVE", "3")))
    except ValueError:
        return 3


# ── Main loop ──────────────────────────────────────────────────────────────────

def run() -> int:
    signal.signal(signal.SIGTERM, lambda *_: _stop.set())
    signal.signal(signal.SIGINT, lambda *_: _stop.set())

    model = _load_model()
    if model is None:
        return 1

    env = _read_env_file()
    threshold = _wake_threshold()
    required_consecutive = _wake_consecutive()

    try:
        import numpy as np
        import sounddevice as sd
    except Exception as exc:
        log.error("Audio stack unavailable (%s) — supervisor cannot run.", exc)
        return 1

    device = _resolve_input_device(env)
    log.info(
        "Supervisor online. Listening for 'wake up rex' "
        "(mic=%s, threshold=%.2f, consecutive=%d, debug=%s).",
        _device_label(device), threshold, required_consecutive, _DEBUG,
    )

    child: Optional[subprocess.Popen] = None
    stream = None
    listening = False
    open_channels = 1  # actual channel count the mic stream was opened with

    # Diagnostics.
    peak_score = 0.0
    last_diag = 0.0
    consecutive = 0  # frames in a row at/above threshold (debounce vs. TV spikes)

    # Channel candidates: the device's real max-input first (e.g. ReSpeaker Lite
    # is 2-in), then mono. macOS PortAudio rejects requesting more channels than
    # the device exposes, and forcing a 2-ch device to 1-ch can yield silence —
    # which is exactly why a "correct" ReSpeaker produced no wake trigger.
    max_in = _device_max_input_channels(device)
    chan_candidates = []
    if max_in and max_in not in chan_candidates:
        chan_candidates.append(max_in)
    if 1 not in chan_candidates:
        chan_candidates.append(1)

    def _open_stream():
        nonlocal open_channels
        last_exc = None
        for ch in chan_candidates:
            try:
                s = sd.InputStream(
                    device=device,
                    samplerate=_SAMPLE_RATE,
                    channels=ch,
                    dtype="float32",
                    blocksize=_CHUNK_SAMPLES,
                )
                s.start()
                open_channels = ch
                if ch != 1:
                    log.info("Mic opened with %d channels (mixing → mono).", ch)
                return s
            except Exception as exc:
                last_exc = exc
                continue
        raise RuntimeError(f"could not open mic with channels {chan_candidates}: {last_exc}")

    def _fire(reason: str):
        nonlocal stream, listening, child
        log.info("Wake detected (%s) — launching controller.", reason)
        if stream is not None:
            try:
                stream.stop(); stream.close()
            except Exception:
                pass
            stream = None
        listening = False
        # Instant audio feedback (chime) before the slower controller boots. Mic
        # is already closed, so the chime can't bleed back into the input.
        _play_chime()
        child = _launch_controller()
        _stop.wait(3.0)  # let main.py take the lock so we don't double-fire

    try:
        while not _stop.is_set():
            running = _controller_running(child)

            # Reap a finished child so the lock check is the single source of truth.
            if child is not None and child.poll() is not None:
                log.info("Controller exited (code=%s). Resuming wake-word listening.", child.returncode)
                child = None
                running = _controller_running(None)

            if running:
                # Dormant: release the mic so the controller owns it, and poll.
                if stream is not None:
                    try:
                        stream.stop(); stream.close()
                    except Exception:
                        pass
                    stream = None
                if listening:
                    log.info("Controller is running — supervisor dormant (mic released).")
                    listening = False
                _stop.wait(1.0)
                continue

            # Active: ensure the mic stream is open and scan for the wake word.
            if stream is None:
                try:
                    stream = _open_stream()
                    model.reset()
                    consecutive = 0
                except Exception as exc:
                    log.error("Could not open mic (%s) — retrying in 2s.", exc)
                    _stop.wait(2.0)
                    continue
            if not listening:
                log.info("No controller running — listening for 'wake up rex'.")
                listening = True

            try:
                audio, _ = stream.read(_CHUNK_SAMPLES)
            except Exception as exc:
                log.warning("Mic read failed (%s) — reopening.", exc)
                try:
                    stream.stop(); stream.close()
                except Exception:
                    pass
                stream = None
                continue

            # Mix to mono: sounddevice returns (frames, channels). On a 2-in
            # device (ReSpeaker Lite) averaging both capsules is what the main app
            # does; a naive reshape would interleave L/R into garbage.
            arr = np.asarray(audio, dtype=np.float32)
            if arr.ndim == 2 and arr.shape[1] > 1:
                samples = arr.mean(axis=1)
            else:
                samples = arr.reshape(-1)
            rms = float(np.sqrt(np.mean(samples ** 2))) if samples.size else 0.0

            # openWakeWord wants int16-range PCM (see _to_oww_input). Feeding raw
            # float [-1,1] pins every score at ~0.001 and nothing ever fires.
            try:
                scores = model.predict(_to_oww_input(samples))
                score = max(scores.values()) if scores else 0.0
            except Exception as exc:
                log.warning("Wake prediction error: %s", exc)
                score = 0.0
            peak_score = max(peak_score, score)
            # Require a short sustained run over threshold, not a single frame —
            # a real phrase holds ~10 frames near 1.0; TV near-misses are 1-2
            # frame spikes. This is the main defense against background-audio
            # false triggers (e.g. firing on the TV).
            if score >= threshold:
                consecutive += 1
                if consecutive >= required_consecutive:
                    _fire(f"onnx score={score:.3f}, {consecutive} consecutive frames")
                    peak_score = 0.0
                    consecutive = 0
                    continue
            else:
                if _DEBUG and consecutive:
                    log.debug("wake run broke at %d frame(s) (score=%.3f < %.2f)",
                              consecutive, score, threshold)
                consecutive = 0

            # ── Periodic diagnostics ───────────────────────────────────────────
            now = time.monotonic()
            if now - last_diag >= 5.0:
                last_diag = now
                log.info("[diag] listening… peak onnx score (last 5s)=%.3f, mic rms=%.4f",
                         peak_score, rms)
                peak_score = 0.0
    finally:
        if stream is not None:
            try:
                stream.stop(); stream.close()
            except Exception:
                pass
        log.info("Supervisor stopping (controller left running: %s).",
                 child is not None and child.poll() is None)

    return 0


# ── Diagnostic modes (no launchd, no controller launch) ────────────────────────

def list_devices() -> int:
    """Print all input devices and which one the supervisor would pick."""
    try:
        import sounddevice as sd
    except Exception as exc:
        print(f"sounddevice unavailable: {exc}")
        return 1
    env = _read_env_file()
    chosen = _resolve_input_device(env)
    try:
        default_in = sd.query_devices(kind="input")
        default_name = str(default_in.get("name", "?"))
    except Exception:
        default_name = "?"

    print("Input devices (index: name):")
    for idx, nm in _list_input_devices():
        marks = []
        if chosen is not None and idx == chosen:
            marks.append("← supervisor will use this")
        elif chosen is None and nm == default_name:
            marks.append("← system default (supervisor will use this)")
        print(f"  {idx:>2}: {nm}  {' '.join(marks)}")
    print()
    nm = (os.environ.get("AUDIO_DEVICE_NAME") or env.get("AUDIO_DEVICE_NAME") or "").strip()
    idx = (os.environ.get("AUDIO_DEVICE_INDEX") or env.get("AUDIO_DEVICE_INDEX") or "").strip()
    print(f".env AUDIO_DEVICE_NAME = {nm!r}")
    print(f".env AUDIO_DEVICE_INDEX = {idx!r}")
    print(f"Resolved mic: {_device_label(chosen)}")
    return 0


def meter(seconds: float = 20.0) -> int:
    """Open the resolved mic and print a live input-level meter PLUS the live
    wakeuprex ONNX score, so you can confirm both that audio is arriving and that
    the model fires on "wake up rex". Speak — the bar should jump and, when you
    say the wake phrase, the score should spike toward 1.0."""
    try:
        import numpy as np
        import sounddevice as sd
    except Exception as exc:
        print(f"audio stack unavailable: {exc}")
        return 1
    env = _read_env_file()
    device = _resolve_input_device(env)
    model = _load_model()
    # Open with the device's real channel count (ReSpeaker Lite = 2-in), mono-mix.
    max_in = _device_max_input_channels(device)
    channels = max_in if max_in else 1
    print(f"Metering mic: {_device_label(device)} ({channels}-ch → mono)")
    print("Speak now — bar = level, score = wakeuprex confidence. Ctrl-C to stop.\n")
    peak = 0.0
    peak_score = 0.0
    try:
        with sd.InputStream(device=device, samplerate=_SAMPLE_RATE, channels=channels,
                            dtype="float32", blocksize=_CHUNK_SAMPLES) as s:
            end = time.monotonic() + max(1.0, seconds)
            while time.monotonic() < end and not _stop.is_set():
                audio, _ = s.read(_CHUNK_SAMPLES)
                a = np.asarray(audio, dtype=np.float32)
                x = a.mean(axis=1) if (a.ndim == 2 and a.shape[1] > 1) else a.reshape(-1)
                rms = float(np.sqrt(np.mean(x ** 2))) if x.size else 0.0
                peak = max(peak, rms)
                score = 0.0
                if model is not None:
                    try:
                        sc = model.predict(_to_oww_input(x))
                        score = max(sc.values()) if sc else 0.0
                    except Exception:
                        score = 0.0
                peak_score = max(peak_score, score)
                bars = int(min(1.0, rms * 20) * 30)
                sys.stdout.write(
                    f"\rrms={rms:0.4f} |{'#' * bars}{' ' * (30 - bars)}| score={score:0.3f}"
                )
                sys.stdout.flush()
    except KeyboardInterrupt:
        pass
    except Exception as exc:
        print(f"\nmic read failed: {exc}")
        return 1
    print(f"\n\nDone. Peak rms={peak:0.4f}, peak wakeuprex score={peak_score:0.3f}")
    if peak < 0.001:
        print("⚠  Essentially silent — the supervisor is NOT getting mic audio.")
        print("   Check Microphone permission for the venv Python and AUDIO_DEVICE_NAME in .env.")
    elif peak_score < _wake_threshold():
        print("⚠  Mic audio is arriving but the wakeuprex score never crossed the")
        print(f"   threshold ({_wake_threshold():.2f}). Say 'wake up rex' clearly into the mic;")
        print("   if it still won't cross, lower REX_SUPERVISOR_WAKE_THRESHOLD.")
    else:
        print("✓  Mic audio is arriving and 'wake up rex' crossed the threshold.")
    return 0


def _print_usage() -> None:
    print(
        "Usage: rex_supervisor.py [command]\n"
        "  (no args)        run the supervisor (listen + launch on wake)\n"
        "  --list-devices   list input devices and show which mic is selected\n"
        "  --meter [secs]   live mic level + wakeuprex score to confirm detection\n"
        "  --test-chime     play the wake chime once and exit\n"
        "  --help           this message\n"
    )


if __name__ == "__main__":
    signal.signal(signal.SIGINT, lambda *_: _stop.set())
    arg = sys.argv[1] if len(sys.argv) > 1 else ""
    if arg in ("--list-devices", "-l", "list"):
        sys.exit(list_devices())
    elif arg in ("--meter", "-m", "meter"):
        secs = float(sys.argv[2]) if len(sys.argv) > 2 else 20.0
        sys.exit(meter(secs))
    elif arg in ("--test-chime", "chime"):
        print(f"Playing chime: {_CHIME_FILE}")
        _play_chime()
        time.sleep(2.0)  # let the fire-and-forget playback finish before exit
        sys.exit(0)
    elif arg in ("--help", "-h", "help"):
        _print_usage()
        sys.exit(0)
    elif arg.startswith("-"):
        print(f"Unknown option: {arg}\n")
        _print_usage()
        sys.exit(2)
    else:
        sys.exit(run())
