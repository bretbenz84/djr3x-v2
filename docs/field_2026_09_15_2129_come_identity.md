# September 15, 21:29: caller uncertainty, approach distance, and continuation

Reviewed the full conversation log and all 1,256 lines of the matching runtime
log, including turn records, motion decisions, recognition diagnostics and errors.

## What happened

* 21:29:31–36: the new silent welcome approach ran at 0.10 m/s and stopped after
  0.51 m at its 60 cm travel budget. This was the intended bounded welcome.
* 21:30:29: “Come here” scored T'Joy .576 and Bret .512, separated by only .065
  against a required .07 margin. The microphone direction estimate was +75°,
  96° from Bret's visible face. Identity correctly abstained instead of assigning
  T'Joy, but the motion locator also abstained. Its fallback demanded a recent
  verified previous speaker, lacked tolerance for the directional contradiction,
  and required a .06 voice-score gap; engagement with the sole visible person
  did not suffice. Rex refused before launching any approach.
* Rex's refusal asked for the exact phrase “Rex, come here.” At 21:30:40 that
  phrase was transcribed with a Bret voice candidate of .586, but the own-echo
  filter dropped it because it matched Rex's recent words. The log cannot prove
  from text alone whether that capture was the human repeat or residual TTS;
  either way, requesting an exact motion phrase created an avoidable echo trap.
* The “who is this one?” at 21:30:43 referred to the animal, not Bret. The animal
  classifier voted cat while Bret's recorded pets were dogs, so the pet-name
  guard asked instead of assuming Max. This is separate from human voice ID.
* 21:31:09: a .642 Bret match and matching direction admitted the next “Come
  here.” That approach travelled 1.03 m. It held when independent front-left
  distance became roughly 0.98–1.04 m, below the then-requested 1.30 m stand-off,
  while camera face-size range still estimated about 3.97 m. It was a distance
  decision, not an identity refusal. The log cannot establish what surface the
  range sensor saw or certify the camera distance as accurate.
* 21:31:28: “Keep going” reached generic conversation because only finite
  turn/move/arc commands had continuation support. It issued no new movement.
* The Max reply also hit `stream_response`'s unguarded first-choice indexing,
  appending the turbulence fallback after useful speech. Weather certificate
  failure and unavailable calibrated compass were additional logged warnings;
  neither explains the identity refusal or the straight approach stand-off.

## Changes

Motion target selection can now use the sole continuously observed engaged
face through a weak voice tie and a conflicting raw direction estimate. It
reuses the independent capture-interval face guard, but treats raw directional
contradiction as uncertainty for LOCATION. Mixed speakers, a second face,
decisive other voice, missing interval evidence, or positively conflicting
mouth/face selection still prevent this fallback. It does not change identity
resolution, assign a name, or authorize voice learning. Ambiguous location
prompts now ask for a visible wave instead of asking who spoke or prescribing
an exact motion command to repeat.

The normal explicit approach now requests 1.00 m rather than 1.30 m stand-off
and runs at up to 0.16 m/s rather than 0.40 m/s. The autonomous welcome retains
its separate 1.30 m stand-off and short travel budget.

“Keep going,” “come closer,” and “a little closer” bind locally to a recent
camera-targeted approach, including after a stand-off hold. They reuse that
target and allow at most 20 cm at 0.08 m/s with a 0.75 m requested stand-off.
Braking margins and the matrix obstacle envelope remain active. Target freshness,
forward bearing, telemetry, room rules and normal motion permissions are checked
before issuing; no user identity is required again. An active approach is not
duplicated. The opportunity expires after 45 seconds or is invalidated by another
issued/refused maneuver, stop or estop. A stale/lost target gets a location
explanation, never an introduction demand. No failed arbitrary turn or unrelated
conversation becomes a forward move.

Classic streaming responses now skip empty-choice chunks, so terminal/usage-only
chunks cannot append the turbulence fallback to an otherwise successful reply.

## Validation and limits

385 tests passed in isolated modules with real hardware/network/audio blocked:
field regressions (7), motion agency (230), arrival (20), come-here regressions
(19), target acquisition (27), streaming timeout (6), solo conversation (11),
verbal confirmation (7), voice learning (46), and greeting identity (12).
New integration cases preserve the recorded weak T'Joy/Bret evidence while
acquiring Bret's visible location, exercise the actual “Keep going” local handler,
permit a slow step using the recorded stop readings, reject missing/close/stale
conditions, preserve stop cancellation, and accept empty terminal stream chunks.

No voiceprints or firmware were changed in this repair. No live drive was used
for verification. Voice matching and range estimates can still be uncertain;
the repair separates those uncertainties from destination selection and makes
closer movement explicitly bounded. Changes load on the next Rex launch.
