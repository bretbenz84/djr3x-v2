# Conversational voice learning

Rex learns CAM++ voices through normal conversation. Saying a name or answering
a confirmation starts a temporary association; it does not save a voiceprint.
There is no voice-ID line to repeat.

## Behavior

- **New person:** Rex's normal name/introduction flow creates or resolves the
  person. Their spoken name answer anchors a temporary acoustic chain. A short
  answer is useful as a reference, but is not counted as a full training sample.
- **Known face without CAM++ prints:** Rex asks, for example, “You on my right —
  Jeff, is that you speaking? I need to recalibrate my audio receptors.” A trusted
  yes/name reply opens collection, followed by an ordinary conversational question.
- **Collection:** at least three accepted utterances, each with two seconds of
  voiced speech and four words, totaling at least eight voiced seconds. Each must
  match the original anchor and accumulated cluster. Padding, repeated capture
  buffers, mixed speakers, competing established voices and untrusted ASR do not
  count. These numerical thresholds are initial defaults requiring live tuning.
- **Several faces:** one person is enrolled at a time. Camera left/middle/right
  describes whom Rex is addressing. The spoken confirmation establishes the mic
  reference; no exact visual angle is projected into a mic bearing. Face tracks
  must remain consistent across capture snapshots. Missing direction is allowed,
  including two-person MacBook tests. Mouth-motion output is never consulted.
- **Direction:** accepted per-utterance ReSpeaker bearings support the acoustic
  chain. A substantially changed bearing or competing direction clusters exclude
  that utterance. Missing DoA never reuses the preceding speaker's bearing. Camera
  or head movement does not rotate the stored microphone reference.
- **Mic mount:** the hero arm moves to `(configured min + max) / 2` and stays
  there. The hold is enforced at the servo write layer, so ordinary animations
  cannot override it. A 1.5-second settling interval excludes movement audio.
  The lease expires after at most 60 seconds even with no new speech. Explicit
  movement/manual control, games, cancellation, sleep and shutdown release it.
  Base motion invalidates the reference. No-servo dev Macs perform no servo I/O.
- **Pause:** insufficient evidence leaves a provisional chain in RAM for up to
  five minutes in the same session. A matching later utterance reacquires a bounded
  hold; collection then resumes after settling. No repeated recitation requests.
- **Existing prints:** retained. Strong existing voice/face agreement can seed
  growth through the same multi-utterance process, up to ten active-model prints.
  Marginal contextual naming cannot refresh a voice or write personal memories.
- **Reply after a greeting/check-in:** if Rex just addressed a known person,
  their enrolled voice is still the best plausible match (the configured known
  speaker floor and margin), and they remain the only visible face throughout
  the capture, Rex continues the exchange without asking their name again.
  This uses the latest unanswered address within 30 seconds, independent face
  snapshots, and no mouth output. An intervening human turn, multiple faces,
  mixed voices or conflicting direction prevents that handoff. It grants only
  conversational context; it neither verifies the voice nor seeds enrollment or
  personal-memory learning. Someone with no CAM++ prints still uses the explicit
  speaking confirmation and sample chain above.

## Recordings and future models

Impersonation now reuses these verified recordings automatically when no saved
clone reference exists. Whole clips and their actual transcripts are combined
at the configured sample rate; at least six voiced seconds are required. The
derived reference lives in the speech cache and is removed when its source voice
data is cleared or merged. This does not modify the recognition prints.

If there is insufficient recorded speech, the explicit impersonation request
asks for ordinary speech in the person's own words. Accepted parts are saved
incrementally under `assets/voices/people/` with completion metadata. Known-person
partials survive cancellation, timeout and restart, so another request resumes
from them. No fixed phrase, recitation matching, or capture-specific echo bypass
remains. Conflicting speakers and mixed/untrusted audio cannot become reference
material. Explicit clone captures do not enroll CAM++ identities.

Accepted float32 WAV audio is stored in the `voice_recordings` table in the local,
gitignored `assets/memory/people.db`. It preserves the capture rate and samples
before model-specific trimming/resampling. This is captured microphone audio,
not an undoing of hardware AEC or microphone processing. Biometric and recording
inserts commit together; a failure cannot leave a half-enrolled voice.

The archive holds up to ten clips per person. Metadata includes capture time,
device, transcript, voiced duration, available bearing, visible identities, model
namespace and acoustic acceptance scores. No rejected sample enters this archive.
Backing up people.db includes the recordings. Merging or deleting a person, clearing
their voice data, or deleting an associated bad biometric also handles the archive.
Explicit exports are independent copies and should be removed separately if no
longer wanted.

```sh
venv/bin/python tools/voice_recordings.py "Jeff Benziger"
venv/bin/python tools/voice_recordings.py "Jeff Benziger" --export
venv/bin/python tools/voice_recordings.py "Jeff Benziger" --reembed campplus
venv/bin/python tools/voice_recordings.py "Jeff Benziger" --reembed campplus --apply
```

Exports go to ignored `assets/memory/voice_exports/<person-id>/`. Re-embedding
previews by default, validates every vector before storage, preserves the original
recordings and all existing model namespaces, and never silently writes a fallback
model as the requested model. This rebuilds speaker embeddings, not model weights.

## Dev-Mac test with Bret and Jeff

1. Restart Rex normally with microphone and camera enabled, outside a game.
2. Keep both people in view. The inspected dev DB had two Bret CAM++ prints and
   no Jeff person row; Jeff should answer the normal name question with his full
   name. If a Jeff face entry is already present on another machine, Rex instead
   asks him to confirm that he is speaking.
3. Let Jeff give several ordinary sentence-length answers. Alternate with Bret;
   Bret's turns should not enter Jeff's sample set. “Yes” alone is not enrollment.
4. Look for `[voice_learning] SAVED ... name='Jeff Benziger'` in the runtime log.
   Rejected/paused turns log their reasons. The recording utility should list
   Jeff's accepted clips only after this save.
5. Continue alternating, then have Jeff speak off camera. Recognition accuracy
   on short/far-field turns still needs this live test; synthetic vector tests
   cannot establish how well the real model separates father and son.

Do not clear Bret's prints or reset the database for this test. Use the normal
startup command, e.g. `venv/bin/python main.py --gui`.

## Implementation and checks

`intelligence/voice_learning.py` owns the chain. `voice_learning_runtime.py` runs
after trust and echo rejection; name handlers contribute only provisional
references. Face history is recorded independently by the face recognition and
optical-tracking loops. `memory/voice_recordings.py` owns atomic archival storage.

The old dictated-sentence, single-clip bootstrap, legacy-voice verification,
passive solo-face storage and separate refresh implementations were removed.
Automatic promotion of anonymous signatures to named people was also removed;
existing named signatures remain readable without opportunistic embedding updates.
Manual `tools/test_voice_id.py` enrollment remains a separate explicit admin tool.

```sh
venv/bin/python tools/run_lean_checks.py voice_learning campplus voice_backend voice_primary_identity voiceless_face_wins dual_intro camping_identity_regression intro_misread_guards jeopardy_evening_run voice_signatures voice_bearing_match voice_direction_quality face_tracking servo_manual_override servo_speech_emotion motion motion_agency
```

The runner uses temporary databases and blocks hardware/audio/network. Physical
servo behavior, live mic direction and conversational recognition need live testing.

## Validation on the dev Mac (2026-09-08)

The focused run passed 777 tests across 29 isolated modules, including 40
conversational-learning tests and the real speech-handler/temporary-SQLite handoff.
Additional admin/body-mood checks reproduced three pre-existing failures against
HEAD sources: two admin fixtures insert legacy `voice` rows while the active model
is CAM++, and one visor fixture assumes neutral 6000 instead of the existing 6560.
No robot, live microphone recording or real API calls were used. Existing local
person records and Bret's two CAM++ prints were inspected read-only and preserved.
