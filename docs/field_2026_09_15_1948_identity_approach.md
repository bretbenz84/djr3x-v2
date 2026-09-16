# September 15, 19:48: identity interruptions and unsafe approach

Sources: `logs/conversation-2026-09-15-19-48-17.log` and matching
`logs/djr3x-2026-09-15-19-48-17.log`. Bret reports he was the only speaker.

## Findings and changes

* At 20:00:52 CAM++ scored T'Joy .550 and Bret .407. The legacy adapter rejected
  the marginal off-camera candidate, but the authoritative resolver accepted
  it as a strong voice. The authoritative owner now rejects that conflict below
  .70. Continuous sole-face evidence can maintain conversational context for a
  plausible enrolled speaker already in the conversation. This requires actual
  independent observations throughout the capture, no mixed speaker evidence,
  no contrary mouth/direction evidence, and no decisive other enrolled voice.
  Contextual attribution explicitly forbids voice learning. Without supporting
  evidence Rex can answer without assigning a name.
* Anonymous voice clusters were counted as separate humans. This turned a solo
  conversation into an apparent group, exposing the stay-quiet tool and the
  background-crosstalk discard path. The long Carter-town story at 19:55:26 was
  discarded that way. Anonymous/uncertain labels no longer establish additional
  people in addressee decisions or the model's participant roster. Actual
  simultaneous faces still establish a group. Background-crosstalk suppression
  now requires a non-solo addressee assessment.
* Generic “What?” could invoke the identity tool. It is now available only for
  an explicit identity-related utterance. The off-camera introduction gate no
  longer interrupts a recently engaged conversation or one with a known face
  present. Reply guidance avoids calling anyone “Guest” or “Unknown,” and
  distinguishes repairing an identity mistake from demanding an introduction.
* “You don't know who I am” hit the incomplete-ending guard on “I am.” Complete
  embedded wh-clauses now pass through; genuinely incomplete endings still hold.
* The 19:59:38 “Back already?” was a deferred startup recent-return greeting,
  using the prior stored visit after minutes of this conversation. Once a person
  has spoken this session, first presence tracking proceeds quietly instead of
  firing the deferred startup greeting. It does not mark a greeting as delivered.
* Idle speech could use an old quiet timer immediately after human content was
  accepted. The conversation-flow gate now also checks the latest content time.

## Close approach

The startup approach began at 19:48:46 with a requested 1.20 m stand-off.
At 19:48:52 its camera range was 3.078 m while front range was 292 mm and
the linear command was still .31 m/s. It held only at 191 mm, then terminated
blocked after travelling 2.59 m. The camera estimate controlled social distance;
front ToF only imposed the 20 cm emergency obstruction gate.

`intelligence/approach.py` now makes front distance a stand-off constraint too,
including speed-dependent braking distance, latency allowance and a settling
latch so braking cannot release into forward creep. It returns blocked when
front stand-off is reached but the caller's camera range remains uncertain.
This applies to every caller of the shared approach controller. Camera range
also prefers vision's normalized face width rather than assuming raw detector
coordinates use the host's default frame width. Logs establish severe camera
overestimation; they do not prove frame scaling was its sole cause.

The prior `0.2.2-tof-quality` firmware repair remains separate: the front-right
radial sensor is physically unreliable and quarantined based on invalid sample
history. These approach changes are Mac-side and do not require another flash.
No physical drive was used for validation. A logged near-obstacle case and a
simulation with a deliberately wrong 5 m camera range exercise braking and
settling at the requested 1.20 m stand-off. Real-world stopping distance still
needs a supervised check with clear space on the next launch.

## Voiceprints and follow-ups

No voiceprints were deleted, relabeled or enrolled during this repair. Bret
confirmed T'Joy said the September 7 enrollment line. A false match in this
session does not prove that enrollment was contaminated. The earlier change
from follow-up age expiration to a delivered-question guard remains in place;
see `field_2026_09_15_sensor_followups.md`.

## Validation

Run modules separately using `venv/bin/python tools/run_lean_checks.py` so real
serial, audio and network access are blocked. Checked voice learning, speaker
segments, primary voice identity, voice bearing, active speaker, target
acquisition, come-here/arrival, addressee, multi-party prompts, tool routing,
greeting identity, identity room, ownership, event follow-ups and presence.
The new field tests drive both authoritative resolution and the actual speech
handler, verify no voice learning, preserve real group handling, and reproduce
the deferred greeting.

Two additional proactive modules retain existing failures:
`lean_memory_musing.test_spoken_musing_sets_once_per_session_flag` and
`proactive_discipline.test_idle_monologue_is_excluded_from_the_cooldown`.
Both were reproduced with the changed source files restored to unmodified HEAD,
then the working edits were restored. The repair takes effect on Rex's next
launch; no live process was restarted or driven for these tests.
