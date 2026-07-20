"""
One process-wide lock serializing ALL MLX (Metal) compute.

Two MLX users share this process: mlx_whisper (transcription) and mlx-audio
Qwen3-TTS (the local voice). MLX is not safe for concurrent graph evaluation
from multiple Python threads — the observed failure is a NATIVE crash, not an
exception: `Fatal Python error: PyThreadState_Get: ... the GIL is released (the
current Python thread state is NULL)` / `Trace/BPT trap: 5`, hit live on the dev
mac (2026-07-19) the moment wake-word transcription overlapped a local-TTS
generation right after boot. The single-threaded POC never saw it; Rex's
speech-queue worker + transcription threads made the overlap possible.

Rule: hold MLX_LOCK around each individual MLX compute call — one whisper
transcribe, one TTS generation step — and NEVER across audio playback or other
blocking I/O. TTS generation acquires it per streamed chunk (~0.3 s each), so a
pending transcription interleaves between chunks instead of stalling for a whole
utterance.

RLock: local_tts's preload warmup generates while already inside its own load
path on the same thread; re-entry must not deadlock.
"""

import threading

MLX_LOCK = threading.RLock()
