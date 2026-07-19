# Local TTS (Qwen3-TTS) + Impersonation Mode — Implementation Plan

Handoff plan for implementing two related features on top of the proven POC in
`~/qwen-tts-test/rex_streaming.py`:

1. **`--local-tts` runtime mode** — run Rex's voice entirely on-device (mlx-audio
   Qwen3-TTS voice clone of the `RX24-pure` reference) instead of ElevenLabs, plus
   **automatic fallback** to the local engine whenever ElevenLabs is unreachable,
   errors, or runs out of credits. ElevenLabs remains Rex's "true voice" and the
   default.
2. **Impersonation feature** — "Rex, do an impersonation of me / of Jimmy Carter":
   Rex clones a voice from a short reference clip + transcript and delivers a short
   LLM-written parody in that voice. Two sources of reference audio:
   a. **People he knows** — captured live (he asks the person to repeat a line),
      saved per-person, and the parody script is mined from that person's memory
      entries (facts/interests/events) for affectionate mockery.
   b. **Famous people** — user-supplied clips + transcripts dropped into a folder.

---

## 0. POC facts to carry over (verified on this machine)

From `~/qwen-tts-test/rex_streaming.py` (read it before starting — it is the
reference implementation for the synthesis/streaming loop):

- Model: `mlx-community/Qwen3-TTS-12Hz-1.7B-Base-8bit` via
  `mlx_audio.tts.utils.load_model`. **RTF ≈ 0.41** (2.5× faster than realtime) —
  best quality/speed tradeoff. `0.6B-Base-bf16` is a lighter alternative;
  `1.7B-Base-bf16` (RTF 1.65) is offline-only quality reference. Keep the variant
  configurable.
- Dependency: `mlx-audio[tts]==0.4.5` (from `~/qwen-tts-test/requirements-rex.txt`).
- Generation API: `model.generate(text=seg, ref_audio=<wav path>, ref_text=<str>,
  stream=True, streaming_interval=0.32)` yields results with `.audio` float chunks.
- Output SR = **24000 Hz** mono float32.
- Long lines are split at sentence boundaries when > **120 chars**
  (`SPLIT_THRESHOLD`, regex `(?<=[.!?—])\s+`) and generated segment-by-segment into
  one playback stream.
- Playback pattern: producer/consumer queue with a **0.25 s pre-roll** buffer
  before opening the `sd.OutputStream`, plus **150 ms front pad** of silence
  (protects against underrun at stream start).
- Rex reference voice: `~/qwen-tts-test/files/RX24-pure.wav` + `RX24-pure.txt`
  (the ref transcript is whitespace-normalized before use).

---

## 1. New module: `audio/local_tts.py`

On-device synthesis engine. Owns the model lifecycle and raw synthesis; it does
**not** own playback parity (LEDs/AEC/gate) — that stays in `audio/tts.py` so both
backends share one parity implementation.

```python
class VoiceRef(NamedTuple):
    wav_path: str
    ref_text: str
    label: str          # "rex", "person:<id>", "famous:<slug>" — for logs/cache keys
```

Public surface:

- `is_available() -> bool` — mlx_audio importable AND model weights present on disk.
- `preload(blocking=True)` — load the model once (thread-safe, idempotent). Log
  load time. Called at boot in `--local-tts` mode; lazily on first use otherwise
  (fallback / impersonation).
- `is_loaded() -> bool`
- `rex_voice_ref() -> Optional[VoiceRef]` — the configured Rex reference
  (`config.LOCAL_TTS_VOICE`, default `RX24-pure`) from `assets/voices/rex/`.
- `generate_stream(text, voice_ref) -> Iterator[np.ndarray]` — sentence-split per
  the POC, yield float32 chunks @ 24 kHz. Ref text whitespace-normalized.
- `synthesize(text, voice_ref) -> tuple[np.ndarray, int]` — buffered convenience
  (concatenate the stream) for cache prefill.

Implementation notes:

- Model load and generation must be serialized behind one lock — MLX generation is
  not assumed re-entrant, and `speech_queue` is single-worker anyway.
- `load_model()` must load from the **local snapshot dir** downloaded by
  `setup_assets.py` (see §6), not trigger a network fetch at runtime. Verify
  early whether `mlx_audio.tts.utils.load_model` accepts a local path; if it
  insists on a repo id, point `HF_HUB_OFFLINE`/`HF_HOME` at the snapshot or use
  `snapshot_download`'s default cache location as the download target. **This is
  the first thing to verify in Phase 1** — it shapes the setup_assets work.
- Failure containment: any exception inside generation logs and raises to the
  caller (`tts.py` decides what to do); never crash the speech worker.

---

## 2. Backend dispatch in `audio/tts.py` (single choke point)

Every spoken line in the app already flows through `tts.speak()` (call sites:
`speech_queue`, `interaction`, `speech_engine`, `main`). Dispatch there — do NOT
touch call sites.

In `speak()`, after the existing no-audio early return and text normalization:

```python
if _local_mode_active() or _api_circuit_open():
    if _speak_local(spoken_text, clean_text, emotion, ...):   # parity path below
        return
    # local failed → fall through to ElevenLabs attempt (best effort both ways)
```

- `_local_mode_active()` — `config.LOCAL_TTS_MODE` (seeded by the `--local-tts`
  flag, §4) and `local_tts.is_available()`.
- **Audio tags:** the local path must synthesize `clean_text` (post
  `strip_audio_tags`) — Qwen would read `[laughs]` literally. Skip
  `_apply_audio_tags` entirely on the local branch. `voice_settings` /
  seed / `previous_text` stitching are ElevenLabs-only concepts: ignored locally.
- `_normalize_for_speech` still applies (WWII → "World War Two" etc.).

### `_speak_local(...)` — playback with full parity

Mirror `_speak_streaming()`'s structure exactly (it is the template — read it
first). Same wrapper obligations, in the same order:

- `output_gate.hold("tts")`, `_speaking` flag, `emotion_orchestrator.frame_for_speech`
  + `publish_frame`, `animations.speech_activity_start`, `servos.begin_speech_motion`,
  `leds_head.speak/ensure_eyes_on`, `leds_chest.speak`, `echo_cancel.set_playing(True)`.
- `gaze_engine.note_about_to_speak` fires in `speak()` before dispatch (already does).
- Open `sd.OutputStream(samplerate=24000, channels=1, dtype="float32",
  **playback_stream_kwargs())`. Buffer the POC's 0.25 s pre-roll before the first
  write; write the 150 ms front pad.
- Per chunk: poll `echo_cancel.was_canceled()` for **barge-in** (abort stream on
  cancel), drive mouth LEDs + `servos.speech_reactive_move` from chunk RMS with the
  `HEAD_LED_SPEAK_LEVEL_MIN_DELTA` throttle — identical to the inline code in
  `_speak_streaming`.
- `conv_log.log_rex(clean_text)` when `log_text`; `on_playback_start` once the
  stream starts; end pad + `stream.stop()`; full `finally` cleanup block
  (LED/servo/animation teardown, `echo_cancel.set_playing(False, tail_secs=...,
  flush=...)`, `_speaking = False`).
- Log `[tts] local synth first audio in X.XXs` (TTFA) and total, tagged
  `backend=local` so `[ttfs]` analysis can distinguish backends.

**Refactor guidance:** `_speak_streaming` and `_speak_local` will share ~60 lines
of parity scaffolding. Extract a shared context-manager/helper (e.g.
`_playback_session(emotion, ...)` yielding a chunk-writer) used by both, rather
than copy-pasting. Keep the diff to `_speak_streaming` behavior-neutral — it is
battle-tested code.

### Caching

- Rex-voice local takes: cache as WAV via the existing `_cache_path` machinery,
  with backend-distinct key inputs: `voice_id=f"local:{voice_label}"`,
  `model_id=config.LOCAL_TTS_MODEL_ID`, no settings/seed/prev tokens. Cache hits
  play through the existing `_read_audio`/`_play` path (works for WAV already).
- **Impersonation takes are NOT cached** — scripts are LLM-generated fresh each
  time; caching only bloats the dir. Gate on `voice_ref.label == "rex"`.
- `is_cached`/`ensure_cached` in local mode key the same way; `ensure_cached`
  in local mode uses `local_tts.synthesize` (used by startup-line prefills).

### Automatic fallback (ElevenLabs → local)

- Config: `LOCAL_TTS_FALLBACK_ENABLED = True` (works even without `--local-tts`,
  provided the model assets exist; if weights are missing, log once and behave as
  today — silence on API failure).
- Trigger points (all funnel to one helper `_fallback_to_local(...)`):
  - `_speak_streaming` request failure / empty stream → today returns False to
    the buffered path; leave that, but…
  - the buffered path's `_fetch_from_api` returning `None` (network error, 401,
    429/quota) → **today the line is silently dropped**. Replace: attempt
    `_speak_local`; only if that also fails, drop.
- **Circuit breaker:** module-level `_api_down_until` timestamp. On any API
  failure set it to `now + LOCAL_TTS_FALLBACK_HOLD_SECS` (default 120). While
  open, `speak()` dispatches straight to local — avoids paying a multi-second
  timeout on every sentence of every reply. Any successful ElevenLabs round-trip
  (including `warmup_api`) clears it. Log transitions:
  `[tts] ElevenLabs down — holding on local voice for 120s` / `[tts] ElevenLabs
  recovered — resuming primary voice`.
- Fallback uses the **Rex** local voice ref, obviously.
- Note: first fallback in a session pays the model-load cost (measure; likely
  several seconds). Acceptable — one delayed line beats a silent robot. Optional
  polish: `LOCAL_TTS_WARM_ON_BOOT = False` config to preload at startup even in
  ElevenLabs mode for users who want instant failover.

---

## 3. Voice reference assets: `assets/voices/`

```
assets/voices/
  rex/RX24-pure.wav          # Rex clone reference (copied from ~/qwen-tts-test/files/)
  rex/RX24-pure.txt
  people/<person_id>.wav     # live-captured impersonation refs
  people/<person_id>.txt     # exact text of the captured line
  people/<person_id>.json    # {name, captured_at, duration_secs} (debuggability)
  famous/<slug>.wav          # user-supplied, e.g. jimmy-carter.wav
  famous/<slug>.txt          # transcript of that clip
```

- **Gitignore the whole `assets/voices/` tree** (add to `.gitignore`). The Rex ref
  is third-party (Disney) audio and person refs are biometric-ish personal data —
  neither belongs in the repo, consistent with the LICENSE's third-party-materials
  note.
- `setup_assets.py` creates the directories (§6). Phase 1 includes a one-time
  `cp ~/qwen-tts-test/files/RX24-pure.{wav,txt} assets/voices/rex/` on this machine.
- Famous-clip matching: slugify the requested name (`"Jimmy Carter"` →
  `jimmy-carter`) and look for `<slug>.wav` + `<slug>.txt`; also accept loose
  matching (all name tokens appear in a slug) so "Carter" or "President Carter"
  finds `jimmy-carter`. An optional `famous/aliases.json` (`{"alias": "slug"}`)
  covers nicknames — nice-to-have, not required for v1.
- A missing `.txt` next to a `.wav` = unusable ref; log a warning naming the file.

---

## 4. `--local-tts` startup flag

Mirror the `--noaudio` env-seed mechanism exactly (`main.py`
`_seed_startup_runtime_flags`):

- `_LOCAL_TTS_ARGS = frozenset({"-local-tts", "--local-tts", "--localtts"})` →
  set `os.environ["DJR3X_LOCAL_TTS"] = "1"` before config imports.
- `config.py`: `LOCAL_TTS_MODE = _env_bool("DJR3X_LOCAL_TTS", False)`.
- `main.py` startup in local mode:
  - Log the voice mode line ("TTS: local Qwen3-TTS (RX24-pure)" vs "TTS: ElevenLabs").
  - **Skip** `tts.warmup_api()` (no point warming a connection we won't use).
  - **Preload** the local model before the startup boot line, analogous to the
    Ollama preload: `local_tts.preload(blocking=True)`. If weights are missing →
    hard, clear startup error telling the user to run `setup_assets.py`
    (mirror the Ollama `OLLAMA_PRELOAD_REQUIRED` failure style).
  - Startup boot/ready lines flow through `tts.speak` as usual and will synth +
    cache locally on first run.
- `--noaudio` continues to short-circuit before any backend dispatch (unchanged).
- README startup-flags table + CONTEXT.md runtime-modes table get a row.

---

## 5. Impersonation feature

### 5.1 Routing: new action `performance.impersonate`

- Add an `ActionSpec` in `intelligence/action_router.py` (category
  `"performance"`, `executable=True`):
  - Description: user asks Rex to do an impersonation/impression of someone, copy
    someone's voice, "talk like me/him/Patrick Stewart", "do me". Args:
    `target` (string: `"me"`, a name, or a famous person's name).
  - It lands in `PERFORMANCE_ACTIONS` automatically via its category → covered by
    the existing performance-request evidence gate
    (`missing_performance_request_evidence`), so a stray "that was a good
    impression" can't trigger it. Verify the gate treats it correctly; add the
    explicit-request evidence check if `performance.*` isn't already wired there.
- Handler lives in a new module `features/impersonation.py` (it's a performance
  feature, like games), called from the action-execution switch in
  `interaction.py`. Keep `interaction.py`'s footprint to: pending-slot plumbing +
  the dispatch call.
- Kill switch: `IMPERSONATION_ENABLED = True` in config; when off, the action
  resolves to a one-line in-character refusal.
- If local TTS assets aren't installed (`local_tts.is_available()` is False), Rex
  declines in character ("my mimicry circuits aren't installed") — pattern:
  `MOTION_NO_BASE_DENIAL_LINES`.

### 5.2 Target resolution (in `features/impersonation.py`)

Given `target`:

1. `"me"`/`"myself"` → the resolved speaker of the turn (`person_id` from the
   identity pipeline). If the speaker is unknown/anonymous → Rex asks who they are
   or offers the capture flow anyway using an anonymous slot? **No — v1 requires a
   known person** (we need their memory entries for the script). Unknown speaker →
   in-character nudge to introduce themselves first.
2. A name matching a known person (`memory/people.find_person_by_name`) → person
   flow (5.3).
3. Else try the famous-clip folder (slug match, §3) → famous flow (5.4).
4. Nothing matches → in-character miss ("I'd need to actually hear them first —
   bring them over or drop me a clip").
   Order matters: known people take precedence over famous clips on a name
   collision.

### 5.3 Person impersonation flow

**If a stored ref exists** (`assets/voices/people/<person_id>.wav` + `.txt`):
skip straight to script generation (5.5). Refresh is out of scope for v1 (a
"capture it again" phrasing can be a follow-up feature; note it in the module
docstring).

**If not — live capture, modeled on the existing `_pending_intro_voice_capture`
pattern in `interaction.py`** (read that flow first; reuse its shape):

1. Rex responds with instructions + a **fixed line for the person to repeat**, so
   the ref text is known exactly rather than depending on transcription. Pick one
   line from `IMPERSONATION_CAPTURE_LINES` (config list, 2 short sentences each,
   ~8–12 s spoken — enough signal for the cloner; the POC's Rex ref is ~3
   sentences). In-character phrasing, e.g.: "Say this like you mean it: 'The
   cantina's open, the music's loud, and I fly better than I sing. Strap in.'"
2. Open pending slot `_pending_impersonation_capture = {person_id, name,
   expected_text, asked_at}` with a timeout (`IMPERSONATION_CAPTURE_TIMEOUT_SECS`,
   ~45 s) and cancel-word handling ("never mind", "forget it") — same hygiene as
   the other `_pending_*` slots.
3. On the next speech segment from that pending slot: `_handle_speech_segment`
   already has `audio_array` (16 kHz mono float32 from the rolling buffer, mic
   pre-roll included). Guards:
   - Speaker must plausibly be the target (same fusion rules the intro capture
     uses); a different confident speaker doesn't consume the slot.
   - Duration ≥ `IMPERSONATION_CAPTURE_MIN_SECS` (default 4.0) — else re-ask once
     (mirror the intro capture's "one more sentence" retry), then give up gracefully.
   - Loose transcript check: ≥ ~50% token overlap with `expected_text` (they may
     paraphrase; the clip still works — prefer saving the **actual transcript**
     over the expected text when they diverge meaningfully. Simplest correct rule:
     save the Whisper transcript as the ref text; it describes what's actually in
     the wav, which is what Qwen needs).
4. Save: write wav (16 kHz is fine if mlx-audio resamples ref audio — **verify in
   Phase 1 by passing a 16 kHz ref to the POC**; if not, resample to 24 kHz with
   `soundfile`/`numpy` on write), write `.txt`, write `.json` sidecar.
5. Proceed to script generation + performance in the same turn: "Got it. Ahem—"
   → impersonation plays.

### 5.4 Famous impersonation flow

Ref = `famous/<slug>.wav` + `.txt`. Script material = the LLM's general knowledge
of the public figure. This is straightforward parody of public figures for a
living-room audience — keep the prompt on the affectionate-roast side
(see 5.5 prompt rules).

### 5.5 Script generation

One non-streaming LLM call (`llm` module, normal conversation model) producing the
parody text — **first person, as the target**, 2–4 sentences, no stage
directions, no audio tags, no quotation marks.

For a known person, assemble material the way `tell_me_about`/person-context
already does: name + relationship, `facts.get_facts(person_id)` (interests,
preferences, running jokes), recent episodic entries. Prompt rules:

- Affectionate roast, as if the person were comically exaggerating themselves —
  their catchphrases, obsessions, known likes. Punch at quirks, not wounds.
- **Hard-exclude** boundary topics and sensitive emotional events (the same
  suppression sets the empathy/boundary layers maintain — reuse whatever helper
  exposes "topics not to joke about" for this person; if none exists as a
  callable, filter facts tagged sensitive + anything from `emotional_events`
  with negative valence).
- Keep it short — this is a bit, not a monologue.

Famous variant: same shape, material from general knowledge, present-tense
self-parody ("mild, witty, no politics-of-the-day cheap shots" steer in the
prompt — it should feel like a cantina lounge act).

### 5.6 Performance delivery

Sequence, all through `speech_queue` so gating/interruption work normally:

1. **Stall/setup line in Rex's own voice** (normal backend), from
   `IMPERSONATION_INTRO_LINES` ("Okay okay — clearing my vocal buffers…"). This
   also covers the model-load latency when the local engine is cold (same trick
   as web search's stall line: the load/generation overlaps playback).
2. **The impersonation** in the cloned voice.
3. Optional **button in Rex's voice** (`IMPERSONATION_OUTRO_ENABLED`, small line
   pool: "I do not sound like that." / "Tip your droid.") — cheap laugh, ship it.

**Plumbing — voice override through the queue:** add an optional
`voice_ref: Optional[VoiceRef]` parameter to `speech_queue.enqueue(...)` (threaded
through `_QueueItem` → the worker's `tts.speak(...)` call) and to `tts.speak(...)`.
Semantics in `speak()`: `voice_ref` present → force the local engine with that ref
(regardless of `--local-tts`; ElevenLabs cannot do this trick), tags stripped,
no caching (per §2). If local synth fails mid-bit, Rex covers in his own voice
("…my impression module just blew a fuse") rather than going silent — enqueue the
cover line from the failure path in `features/impersonation.py`, not from inside
`tts.py`.

Memory: record an episodic entry ("I did an impersonation of Bret — nailed it.")
via the existing episodic capture API, consistent with how games/celebrations log.

---

## 6. `setup_assets.py` + requirements

- **`download_qwen_tts_model(root)`** following the existing
  `download_ecapa_model`/`download_whisper_model` pattern:
  - `snapshot_download` of `mlx-community/Qwen3-TTS-12Hz-{LOCAL_TTS_MODEL_VARIANT}`
    into `assets/models/qwen_tts/<variant>/` (gitignored like the other model
    dirs), `config.json` as the fully-downloaded sentinel, skip when present,
    report created/skipped/failed like the others. ~2 GB for 1.7B-8bit — print
    the size warning the way the InsightFace/RF-DETR entries do.
  - Wire into `main()`'s download sequence and the summary report.
  - Per §1: confirm `mlx_audio.load_model` loads from this local dir; if it
    fights you, download into the default HF cache instead and record the
    decision in the module docstring.
- **Directories:** add `assets/voices/rex`, `assets/voices/people`,
  `assets/voices/famous` to the `create_directories` list.
- **Requirements:** add `mlx-audio[tts]==0.4.5` to `requirements.txt` in the
  Apple-Silicon ML section. **Verify the resolved `mlx` version satisfies both
  `mlx-whisper>=0.4.0` and mlx-audio 0.4.5 in the venv before committing** —
  `pip install` in the venv, then run the existing test suite plus a whisper
  transcription smoke check. If they conflict, pin whatever compatible pair the
  POC venv proves out (`~/qwen-tts-test` has a working install to compare
  against: `pip freeze | grep -i mlx` there).
- `setup_macos.sh`: nothing new needed beyond what setup_assets already gets.

---

## 7. Config additions

`config.py` (internal defaults):

```python
# ── TTS — LOCAL (Qwen3-TTS on-device clone) ─────────────────────────────
LOCAL_TTS_MODE = _env_bool("DJR3X_LOCAL_TTS", False)   # seeded by --local-tts
LOCAL_TTS_MODEL_VARIANT = "1.7B-Base-8bit"             # RTF 0.41 on this machine
LOCAL_TTS_MODEL_ID = f"mlx-community/Qwen3-TTS-12Hz-{LOCAL_TTS_MODEL_VARIANT}"
LOCAL_TTS_MODEL_DIR = "assets/models/qwen_tts"
LOCAL_TTS_VOICE = "RX24-pure"                          # assets/voices/rex/<voice>.{wav,txt}
LOCAL_TTS_SAMPLE_RATE = 24000
LOCAL_TTS_SPLIT_THRESHOLD = 120
LOCAL_TTS_STREAMING_INTERVAL = 0.32
LOCAL_TTS_PREROLL_SEC = 0.25
LOCAL_TTS_FRONT_PAD_MS = 150
LOCAL_TTS_FALLBACK_ENABLED = True                      # ElevenLabs failure → local voice
LOCAL_TTS_FALLBACK_HOLD_SECS = 120                     # circuit-breaker hold
LOCAL_TTS_WARM_ON_BOOT = False                         # preload even in ElevenLabs mode

# ── IMPERSONATION ───────────────────────────────────────────────────────
IMPERSONATION_ENABLED = True
IMPERSONATION_VOICES_DIR = "assets/voices"
IMPERSONATION_CAPTURE_MIN_SECS = 4.0
IMPERSONATION_CAPTURE_TIMEOUT_SECS = 45.0
IMPERSONATION_CAPTURE_LINES = [...]                    # lines Rex asks the person to repeat
IMPERSONATION_INTRO_LINES = [...]                      # Rex-voice setup/stall lines
IMPERSONATION_OUTRO_ENABLED = True
IMPERSONATION_OUTRO_LINES = [...]
```

`user_config.example.py` (user-facing, commented-out at defaults, per its
conventions — remember existing users must copy new sections manually):
`LOCAL_TTS_FALLBACK_ENABLED`, `LOCAL_TTS_MODEL_VARIANT`, `LOCAL_TTS_VOICE`,
`IMPERSONATION_ENABLED`, `IMPERSONATION_CAPTURE_LINES`, `IMPERSONATION_INTRO_LINES`,
`IMPERSONATION_OUTRO_*`. (`--local-tts` itself is a flag, not a config override.)

---

## 8. Tests (`unittest`, mlx_audio mocked — no model load in CI)

New `tests/test_local_tts.py`:
- Dispatch: `LOCAL_TTS_MODE=True` routes `speak()` to the local path; ElevenLabs
  client never touched.
- Tags: local path receives tag-stripped text (feed a line with `[laughs]`).
- Fallback: `_fetch_from_api` → None triggers local synth; circuit breaker opens,
  routes subsequent lines local without touching the API, and closes after a
  simulated success.
- Cache keys: local Rex-voice key ≠ ElevenLabs key for identical text; `voice_ref`
  (impersonation) takes are not cached.
- No-audio mode still short-circuits before local dispatch.
- `is_available()` false (import error / missing weights) → behavior identical to
  today.

New `tests/test_impersonation.py`:
- Target resolution: "me" → speaker; known-person name; famous slug (incl. loose
  match "Carter" → `jimmy-carter`); known person beats famous on collision;
  unknown → refusal line.
- Capture slot: opens with expected text; short clip re-asks once; good clip
  writes wav/txt/json (tmp dir) and proceeds; timeout and cancel-words clear it;
  wrong confident speaker doesn't consume it.
- Script-gen prompt excludes boundary/sensitive material (assert the prompt
  assembly, LLM mocked).
- `IMPERSONATION_ENABLED=False` and missing local engine → refusal paths.
- Action gate: `performance.impersonate` blocked without explicit-request evidence.

Existing suites must stay green: `venv/bin/python -m unittest discover -s tests`.

**Live verification is participatory** — model load timing, TTFA, clone quality,
barge-in, fallback behavior with network pulled: **ask Bret before running
anything that speaks or needs him to talk** (house rule). Passive checks
(imports, model load timing in isolation) are fine.

---

## 9. Phasing

**Phase 1 — local engine + `--local-tts`** (foundation)
1. Verify: mlx-audio + mlx-whisper coexistence in the venv; `load_model` from a
   local dir; 16 kHz ref-audio acceptance (POC one-liner tests).
2. `setup_assets.py` download + dirs; requirements; copy RX24-pure into
   `assets/voices/rex/`; gitignore.
3. `audio/local_tts.py`; parity-refactor + `_speak_local` in `audio/tts.py`;
   `--local-tts` flag + config + main.py preload/skip-warmup.
4. Tests; then a live run (`python main.py --local-tts`) with Bret.

**Phase 2 — automatic fallback** (small, high value)
`_fallback_to_local` + circuit breaker + `ensure_cached` local path + tests.
Live test: kill network mid-session (with Bret).

**Phase 3 — impersonation**
`features/impersonation.py`, `performance.impersonate` ActionSpec + gate wiring,
capture pending-slot in `interaction.py`, script generation, famous-folder
loader, `speech_queue`/`tts.speak` `voice_ref` threading, episodic logging,
tests.

**Phase 4 — polish/docs**
`user_config.example.py` sections, README (features bullet + flags table),
CONTEXT.md (runtime modes, repo map entries for `audio/local_tts.py` and
`features/impersonation.py`), this doc updated to "as-built".

Commit convention: verified work goes straight to `main` (house rule), one commit
per phase or coherent sub-step.

## 10. Open questions / risks (resolve during Phase 1, don't block on them now)

- `load_model` local-path support (§6) — shapes the download target.
- mlx version compatibility between mlx-whisper and mlx-audio (§6).
- Model load time + RAM alongside Whisper/InsightFace/RF-DETR/ECAPA — measure;
  if load is >10 s it strengthens the case for the stall-line covers already
  planned.
- 16 kHz mic capture as ref audio (§5.3) — else resample on save.
- GIL pressure: Qwen generation is compute-heavy; confirm playback stays smooth
  (deep-buffer `playback_stream_kwargs` should cover it, same as model preloads
  at boot) and that camera/wake-word threads don't starve during a long bit.
- Qwen quality on very short lines (one-word "Yes." replies in local mode) —
  if clipped/odd, consider a minimum-length pad or accept the quirk.
