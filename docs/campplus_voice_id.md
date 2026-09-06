# CAM++ voice identification

CAM++ is the default voice embedder as of 2026-09-06. Set `VOICE_EMBEDDER` in
`config.py` to `"campplus"`, `"ecapa"`, or `"resemblyzer"`, then restart Rex.
The `VOICE_EMBEDDER` environment variable can override the default.

## Model and installation

The pinned checkpoint is the Chinese/English advanced CAM++ model from
[3D-Speaker / ModelScope](https://www.modelscope.cn/models/iic/speech_campplus_sv_zh_en_16k-common_advanced/summary),
using the [sherpa-onnx ONNX export](https://github.com/k2-fsa/sherpa-onnx/releases/tag/speaker-recongition-models).
It runs through **ONNX Runtime on CPU**, with two inference threads by default.
This implementation does not require MLX conversion or a new STT backend.

```sh
venv/bin/python tools/download_campplus.py
```

`setup_assets.py` also downloads it. The 28,281,164-byte model lives at
`assets/models/campplus/campplus.onnx`; model assets are ignored by Git, so run the
command on another robot/dev Mac after pulling. SHA-256:

```
aa3cfc16963a10586a9393f5035d6d6b57e98d358b347f80c2a30bf4f00ceba2
```

Downloads and loading validate this checksum. Missing or invalid CAM++ weights
log an actionable error and disable voice embeddings; they never silently load
ECAPA into the CAM++ storage space. Existing `onnxruntime`, `torch`, `torchaudio`,
`numpy`, and `scipy` dependencies supply inference and feature extraction.

Audio is resampled to 16 kHz, capture padding is trimmed, and 80-bin Kaldi
filterbanks are computed on normalized floating-point samples, followed by
per-bin time-mean subtraction. Embeddings are normalized 192-dimensional vectors.
The model can process short speech; the current input floor is 200 ms. This is
not a guarantee of correct identification on every short word or vocalization.

## Existing people and automatic enrollment

Old embeddings cannot be converted into CAM++ embeddings without the original
audio. Legacy prints remain intact for rollback. CAM++ prints use biometric type
`voice_campplus_zh_en_v1`, and anonymous signatures use `voice_signatures_campplus`.
Both CAM++ and ECAPA output 192 floats, so all matching/counting paths separate
models by storage namespace as well as dimension. Existing DBs gain the signature
table through the normal migration path; no old person or print is deleted.

With `CAMPPLUS_AUTO_ENROLL_ENABLED = True`, a missing CAM++ profile is created
from a trusted spoken turn when either:

- Interval mouth-motion evidence consistently identifies one existing person
  (at least three observations at confidence 0.5 or higher); or
- The speaker explicitly identifies themselves, such as “My name is Bret,”
  and that name resolves to an existing person without conflicting visual evidence.

The first enrollment requires at least one second of voiced audio and the
existing 1.2-second capture minimum. Recognized mixed-speaker captures, conflicting
visual speakers, low-confidence transcripts, laughter, and typed GUI text cannot
seed this profile. After saving, the same turn is rescored so it can immediately
carry the speaker's name. Established enrollment/introduction flows also use CAM++.
Opportunistic profile growth additionally requires interval active-speaker evidence.

A visible face or prior conversation partner alone is insufficient. If Rex cannot
establish who an un-enrolled voice belongs to, it remains unknown until identity
is established. A new model cannot recover that name from old incompatible vectors.

## Validation and morning test

Offline regression checks:

```sh
venv/bin/python tools/run_lean_checks.py campplus voice_backend voice_signatures speaker_id_margin speaker_segments voice_primary_identity camping_identity_regression
```

The CAM++ tests exercise the real local ONNX model's input/output contract,
separate storage for same-dimension legacy prints, rollback, first-turn enrollment
and attribution, repeated recognition, and enrollment rejection conditions.
Synthetic audio checks establish inference behavior, **not recognition accuracy**.

`CAMPPLUS_MATCH_THRESHOLD = 0.50` and `CAMPPLUS_MATCH_MARGIN = 0.07` are initial
raw-cosine defaults, pending live validation. ECAPA's historical +0.25 offset and
thin-runner margin relaxation do not apply to CAM++. Other conservative learning
and unknown-signature thresholds remain in effect.

Tomorrow, have each person speak a clear sentence while facing Rex; if their
identity is not established, use “My name is …” once. Then alternate short replies,
including immediate back-to-back turns, followed by off-camera speech. Check names
in HEARD lines and the `[campplus]` enrollment / `[speaker_id]` scoreboards. No
microphone capture, speaker playback, or robot movement is needed by the offline
CAM++ tests.
