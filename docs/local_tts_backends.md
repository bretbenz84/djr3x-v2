# Local speech: Breeze TTS 2 and Qwen3-TTS

Breeze TTS 2 **8-bit** is the default local voice engine as of 2026-09-09.
ElevenLabs remains the normal online voice. The selected local engine serves
`--local-tts`, automatic offline mode, ElevenLabs error/quota fallback, and all
impersonations. Breeze 4-bit, mixed precision and 16-bit are not used.

## Configure and install

Set in `user_config.py`, then restart Rex:

```python
LOCAL_TTS_BACKEND = "breeze"  # default; use "qwen" for the previous engine
```

Install dependencies and the selected model:

```bash
venv/bin/python -m pip install -r requirements.txt
venv/bin/python setup_assets.py
venv/bin/python main.py --local-tts
```

The Breeze integration requires `mlx-audio[tts]==0.5.1`. Its depth-cache adapter
uses that release's internal API, so upgrading the library requires rechecking
the adapter. On this Mac the upgrade also changed transformers 5.12.1 → 5.16.1
and tokenizers 0.22.2 → 0.23.2; `pip check` passes. MLX stayed at 0.32.0.

Setup downloads only the selected local engine:

| Engine | Snapshot | Location |
| --- | --- | --- |
| Breeze | `mlx-community/Breeze-TTS-2-mlx-8bit`, ~4.6 GB | `assets/models/breeze_tts/8bit/` |
| Qwen | `mlx-community/Qwen3-TTS-12Hz-1.7B-Base-8bit`, ~2.9 GB | `assets/models/qwen_tts/1.7B-Base-8bit/` |

Breeze is pinned to snapshot `c6e4a2ff6ab9afba68b7853de802273ffe23fb49`, matching
the test bench. Setup fetches the entire snapshot, including `audio_tokenizer/`;
missing model weights, codec weights or tokenizer files cannot count as a
completed installation. Runtime loads by absolute project path with Hugging Face
network loading disabled. Weights remain gitignored. Existing Qwen assets are
kept when switching engines. The initial Breeze installation was copied from the
bench's Hugging Face cache, with symlinks resolved to real files.

Breeze uses `BREEZE_TTS_VOICE="RX24-pure-24k"`, the bench's matched 24 kHz mono
reference and exact transcript, tracked under `assets/voices/rex/`. Qwen keeps
`LOCAL_TTS_VOICE="RX24-pure"`. Famous and captured-person reference pairs work
with either engine: pass the WAV **path**, not an unresampled array. Never pair
a shortened WAV with the original full transcript.

`--local-tts` changes speech synthesis only; it does not force the reply brain
local. Automatic offline mode separately routes replies to Ollama `qwen3.5:2b`
and skips hosted features. Install that brain with `ollama pull qwen3.5:2b` if
needed. Local TTS startup currently logs missing/broken assets and permits
ElevenLabs fallback instead of terminating, so `--local-tts` is not a strict
network prohibition when installation is broken.

## Streaming and playback

`audio/local_tts.py` retains one public interface for both engines. Breeze uses
two optimizations ported from `/Users/bbenziger/Local/breeze-tts-2/scripts/`:

- `audio/breeze_fast_depth.py`: incremental depth-decoder KV caching rather than
  repeatedly recomputing the growing codebook prefix. The bench found numerical
  differences on 8-bit; this is not claimed to be bit-identical to stock.
- `audio/breeze_ref_cache.py`: reference-prefix embeddings cached per model,
  speaker, exact transcript and reference path/mtime/size. An eight-entry LRU
  bounds memory across impersonations; changed recordings invalidate the key.

Breeze generates one continuous utterance, including for impersonations. A
background `Take` prepares raw 24 kHz mono float32 audio. For Rex's ordinary
voice, it streams chunks through a bounded queue; `first_ready` means one chunk
is available. For every person/famous/anonymous impersonation, it collects the
complete utterance before publishing one playable unit. `first_ready` then
means synthesis and generator cleanup have finished. The model's token cap
bounds the one-shot audio in memory; generated impressions are never cached.
Cancellation stops the producer and closes its generator on the producer thread,
releasing synthesis ownership. MLX compute and teardown still share the existing
process-wide MLX lock with ASR.

Breeze starts after the intro/reply is complete, even online, because that line
could itself fall back to the same local engine. A parked stream holding a full
queue must not block the intro waiting for the engine. Qwen keeps its existing
buffered impression path. The `LOCAL_TTS_TAKE_*` buffering switches and
`LOCAL_TTS_CLONE_FULL_BUFFER` apply to Qwen, not Breeze.

`audio/tts.py` preserves the output gate, AEC, delivery checks, cancellation,
mouth pacing and speech-motion cleanup. Ordinary Breeze speech uses 1.5 seconds
of preroll. All Breeze playback requests a 0.35-second host buffer and 4096-sample
blocks. Impersonations keep the thinking loop running while the full take is
prepared, trading a longer initial pause for continuous playback.

The 21:55:46 field log showed an impression being deliberately aborted after
three PortAudio underruns. That abort rule and its configuration were removed:
an underrun reports earlier buffer starvation, and subsequent writes must still
be drained. Full preparation keeps clone inference out of the playback window.
Actual device errors and cancellation still stop playback. Failed or timed-out
preparation never publishes a partial Breeze impression, and an explicit
performance checks the delivery receipt before its success outro/episode.

| Breeze setting | Default | Meaning |
| --- | --- | --- |
| `BREEZE_TTS_STREAMING_INTERVAL` | `0.25` s | Requested generation chunk interval |
| `BREEZE_TTS_PREROLL_SEC` | `1.5` s | Ordinary Rex speech preroll; clones prepare fully |
| `BREEZE_TTS_FRONT_PAD_MS` | `0.0` | No extra leading silence |
| `BREEZE_TTS_OUTPUT_LATENCY` | `0.35` s | PortAudio host-buffer request |
| `BREEZE_TTS_OUTPUT_BLOCKSIZE` | `4096` | Device block size in samples |
| `BREEZE_TTS_QUEUE_CHUNKS` | `16` | Maximum queued chunks ahead of ordinary Rex speech |
| `BREEZE_TTS_TEMPERATURE` | `0.7` | Bench's steadier sampling setting |
| `BREEZE_TTS_MAX_TOKENS` | `750` | Absolute generation cap |
| `BREEZE_TTS_DURATION_SLACK` | `2.0` | Additional word-count-based duration cap |

Generation uses `cfg_scale=1.0` to avoid doubling inference work. No voice-direction
prompt is added by this migration. Optional synthesized-output caching remains
off by default; cache identities distinguish engine/model/reference so a switch
cannot replay an older voice. Impersonation output is never cached. Reference
prefix caching caches conditioning only, not speech output.

Ordinary streamed speech can still underrun on a busy or thermally constrained
Mac. Increasing `BREEZE_TTS_PREROLL_SEC` or the host buffer trades start latency
for continuity. Streaming continues after preroll; short lines play once fully
generated if they end before reaching the buffer threshold. The revised buffers
still need live validation and cannot compensate indefinitely if generation
runs slower than playback. The bench's 8-bit choice is
retained regardless of load; there is no automatic downgrade to 4-bit.

## Validation

Isolated regression checks (no GPU, network, audio or serial):

```bash
venv/bin/python tools/run_lean_checks.py breeze_tts local_tts impersonation_take impersonation organic_impersonation offline_mode tts_network_resilience clone_deep_buffer streaming_tts two_chunk_tts tts_led_cleanup
```

The Breeze tests prove Rex speech starts after six 0.25-second chunks while
generation continues, and person/famous impressions wait for the complete audio.
They verify every sample survives repeated reported underruns, including a
distinctive ending, plus cancellation during preparation, partial-generation
failure, preparation timeout and unsuccessful delivery. Existing checks cover
full-queue cancellation, parked-take replacement, backend selection, offline/API
fallback, cache separation and incomplete assets. Legacy Qwen fixtures select
Qwen explicitly.

Offline synthesis benchmark, writing WAV/JSON files without audio playback:

```bash
venv/bin/python tools/bench_local_tts.py --backend breeze --out-dir /tmp/rex-breeze
venv/bin/python tools/bench_local_tts.py --backend qwen --out-dir /tmp/rex-qwen
```

This needs Metal GPU access. It blocks network connects, loads only local assets,
and makes two calls per reference to expose initial and warmed-reference costs.
On the development Mac, the final matched-reference Breeze run measured:

| Voice / call | First chunk | Additional head start for continuous playback | Estimated continuous start |
| --- | --- | --- | --- |
| Rex 1 | 0.317 s | 0.226 s | 0.544 s |
| Rex 2 | 0.463 s | 0.035 s | 0.499 s |
| Jimmy Carter 1 | 0.982 s | 0.283 s | 1.265 s |
| Jimmy Carter 2 | 0.681 s | 0.101 s | 0.782 s |

Load plus first-chunk warmup took 3.24 s in this run. Earlier runs had slower
initial calls. These are four short synth-to-file samples, **not** live robot or
device latency guarantees. Qwen also completed offline synthesis for both voices
under mlx-audio 0.5.1. No speaker playback or robot controller was started.
Live listening, device underruns, long impressions and competing ASR/LLM load
still need validation on the intended machine.

The [Breeze model card](https://huggingface.co/mlx-community/Breeze-TTS-2-mlx-8bit)
links the research/non-commercial weight license. The copied snapshot retains its
LICENSE and NOTICE files. See `local_tts_impersonation_plan.md` for the historical
Qwen design; current Breeze preparation and playback are described above.

The timing table above predates the live buffering correction. First-chunk
generation time is not the new time to audible speech: preroll and the device
buffer intentionally add latency to reduce stuttering.
