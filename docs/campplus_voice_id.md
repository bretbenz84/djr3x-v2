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
from a trusted spoken turn using any of these independent identity sources:

- One actually visible named face agrees with the strongest legacy voice match.
  With `CAMPPLUS_LEGACY_BOOTSTRAP_ENABLED`, the captured audio is temporarily
  verified against the person's old ECAPA or Resemblyzer prints. This needs two
  seconds of voiced speech, a conservative raw-cosine threshold, and a margin
  over other people. It never changes the active CAM++ backend or copies an old
  vector into CAM++ storage. Once that person has a CAM++ print this path stops.
- The speaker explicitly identifies themselves, such as “My name is Bret,”
  and that name resolves to an existing person without conflicting visual evidence.
  A complete existing name, such as “Bret Benziger,” also qualifies when that
  named person is the sole visible face. Ordinary sentences cannot create people.
- Interval mouth-motion evidence consistently identifies one existing person
  (at least three observations at confidence 0.5 or higher), when available.

Mouth detection is optional. The owner reports that it has never worked reliably
on these cameras/models, so enrollment and profile growth cannot depend on it.

The first enrollment requires at least one second of voiced audio and the
existing 1.2-second capture minimum. Recognized mixed-speaker captures, conflicting
visual speakers, low-confidence transcripts, laughter, and typed GUI text cannot
seed this profile. After saving, the same turn is rescored so it can immediately
carry the speaker's name. Established enrollment/introduction flows also use CAM++.
Opportunistic profile growth requires an existing strong CAM++ match and agreement
with the sole currently visible named face; it does not require mouth detection.

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


## 2026-09-06 dev-Mac enrollment failure

The 11:10:20 session loaded CAM++ and recognized Bret's face, but the copied DB
contained six old ECAPA prints for Bret and zero CAM++ prints for anyone. Short
windows from his single-speaker speech had cosines as low as 0.134; the original
integration incorrectly used the 0.50 identity threshold as a speaker-change
threshold. It declared six turns mixed and blocked enrollment. No usable active
mouth identity was logged; the bare full-name reply also missed the self-ID parser.

CAM++ window cosines now remain diagnostics (`acoustic_change_suspected`), not
proof of another speaker. Distinct positively identified window speakers and
conflicting visual speakers still block whole-clip enrollment/attribution and can
support splitting at actual silent gaps. Unenrolled-speaker diarization remains
unvalidated; low cosine alone cannot establish it. Legacy backends retain their
previous window policy.

Every CAM++ enrollment attempt records its outcome, voiced duration, available
visual evidence, and legacy verification scores in `[campplus] enrollment` and
the turn trace's `campplus_enrollment` field. The copied DB and logs are evidence;
this repair does not manufacture voiceprints without captured audio or modify them.

## 2026-09-06 short replies and duplicate enrollment

The 11:45:56 dev-Mac run successfully enrolled Bret. Its two CAM++ rows were
byte-identical: initial enrollment and automatic refresh saved the same capture.
CAM++ insertion now reuses an identical existing print, and counts/centroid matching
ignore historical exact duplicates. No existing biometric rows are deleted.

With the owner's approval, `CAMPPLUS_SHORT_REPLY_CONTINUITY_ENABLED = True` allows
conversational attribution for replies of at most four words and 1.5 voiced seconds.
It requires the same recent verified speaker (within 90 seconds), a matching raw
voice candidate at least `CAMPPLUS_SHORT_REPLY_MIN_COSINE = 0.20`, sufficient runner-up
margin, and at least three capture-interval observations each showing only that
person's visible face. Conflicting direction, mixed speech, or another visible face
blocks continuity. Mouth-motion detection is not required.

These verdicts carry `learning_allowed=False`: they cannot refresh voiceprints or
add personal memories. They also cannot extend the verified-speaker timestamp;
an intervening unidentified turn clears continuity. The strong CAM++ threshold
remains 0.50. This is contextual naming, not proof that CAM++ alone identified a
short utterance. The logged .490, .414, and .234 cases have regression coverage
with explicit interval evidence, plus rejection cases where that evidence is absent.

“Do you know who's speaking?” now uses the authoritative verdict directly:
“You're Bret Benziger,” or “I think you're Bret Benziger” for contextual attribution.
An unresolved verdict says Rex is uncertain. A second LLM paraphrase no longer
turns the identified human into Rex saying “it's me.”

## Room capture in the same session

Naming a room starts visual collection; it does not prove recognition or a saved
memory. Acknowledgments now reflect that distinction, and conversation grounding
retains a recent user-reported room name instead of asking for it again.
Repeated person occlusion triggers one request to step aside; skipped capture
reasons and timeout counts are logged. Off-image pose landmarks are clamped to the
image before estimating occlusion, and enrollment times out even without camera
frames. A failed capture says the visual memory was not saved, without asking for
the room's name again. The old log recorded zero collected frames but did not log
why each was rejected; the original cause cannot be proved retrospectively.

Run the offline regressions with:

```sh
venv/bin/python tools/run_lean_checks.py dev_mac_identity_room campplus voice_primary_identity place_recognition place_questions
```

Live alternating-speaker accuracy and room recognition still require validation
on the robot. The copied dev-Mac database and logs remain unchanged.
