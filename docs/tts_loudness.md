# ElevenLabs speech loudness

ElevenLabs speech is leveled at playback by default. This covers fresh streamed
PCM, prefetched MP3 continuations, and existing MP3/WAV cache hits. Local Breeze
voices and impersonations retain their existing playback behavior.

`audio/speech_level.py` runs the same incremental processor on complete takes
and network chunks. It measures K-weighted energy in overlapping 400 ms windows
on a 20 ms clock, excludes silence, and gradually adjusts gain toward -20 LUFS.
It uses the weighting filters from [ITU-R BS.1770](https://www.itu.int/rec/R-REC-BS.1770).
This is a causal speech leveler, not an exact whole-program integrated-LUFS
normalizer: short interjections, emphasis and peak limiting still cause some
variation.

A 300 ms audio pre-roll sets initial gain before opening the live output stream.
The actual added wall-clock delay depends on how quickly ElevenLabs supplies
those samples; it does not wait for the entire line. A limiter inspects 4×
interpolated peaks, looks one 20 ms frame ahead, and ramps attenuation around
transients with a 100 ms release. Its nominal ceiling is -3 dBFS, with headroom
for reconstruction peaks. Gain is held through quiet gaps and capped at +26 dB.

Cache files retain their original gain. New streamed WAVs also retain the raw
tail; cached playback trims trailing silence after leveling so the fixed trim
threshold does not remove quiet words. No cache migration, deletion, or API
regeneration is needed. The mouth and speech-reactive motion receive the same
processed samples as the speaker. Cancellation discards pending pre-roll and
limiter samples, closes the API stream, and avoids caching an interrupted take.

An explicit `[whispers]` tag lowers the target by 4 dB for that synthesized
chunk, including any ordinary text in the same chunk. It preserves quieter
delivery without leaving the recording at an unintelligible level.

## Configuration

These values are in `config.py` and can be overridden in `user_config.py`:

| Setting | Default | Meaning |
| --- | ---: | --- |
| `TTS_LOUDNESS_ENABLED` | `True` | Enable ElevenLabs playback leveling |
| `TTS_LOUDNESS_TARGET_LUFS` | `-20.0` | Rolling speech loudness target |
| `TTS_LOUDNESS_MAX_GAIN_DB` | `26.0` | Maximum boost for quiet speech |
| `TTS_LOUDNESS_PEAK_DBFS` | `-3.0` | Limiter ceiling |
| `TTS_LOUDNESS_PREROLL_MS` | `300.0` | Initial audio analysis buffer |
| `TTS_LOUDNESS_WHISPER_OFFSET_DB` | `-4.0` | Target offset for tagged whispers |

Restart Rex after code or configuration changes. Setting
`TTS_LOUDNESS_ENABLED=False` restores original-gain ElevenLabs playback.

## September 22, 2026 validation

The 50 most recently written cache files were decoded and processed offline;
their original files were not edited, and nothing was played on the robot.
FFmpeg `loudnorm` measured integrated LUFS and true peak before/after. All output
durations and sample rates were preserved.

- The latest session's 32 new clips went from a **24.69 dB** loudness spread
  (-44.65 to -19.96 LUFS) to **4.90 dB** (-22.99 to -18.09 LUFS).
- Across all 50 clips, spread fell from **27.41 dB to 5.32 dB**.
- The quiet “You can take the rest of the night off…” take rose from
  **-44.65 to -20.87 LUFS**.
- “That tracks” and its continuation went from **7.75 dB apart to 0.10 dB**.
- Maximum measured output true peak across all 50 was **-2.85 dBTP**.

Per-file measurements: [tts_loudness_validation.csv](tts_loudness_validation.csv).
The listening result and real API pre-roll latency still need a normal robot
session; the offline checks do not establish those subjective/live properties.

Regression checks run with hardware, network and audio blocked:

```sh
venv/bin/python tools/run_lean_checks.py speech_level tts_tail tts_pronunciation delivery_contract shared_device_guard tts_led_cleanup two_chunk_tts streaming_tts tts_network_resilience
```

The tests cover arbitrary PCM byte/chunk boundaries, cached/streamed sample
parity, original-cache preservation, silence and pauses, gain/peak bounds,
whispers, short-clip flushing, pre-roll failure, interruption, and mouth samples.
