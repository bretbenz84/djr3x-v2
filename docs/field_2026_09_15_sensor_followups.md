# September 15: front-right sensor and follow-up delivery

## Firmware deployed

Flashed and handshook `0.2.2-tof-quality`, using the Bluepad32 ESP32 core,
115200 upload baud, and all four existing hardware flags (drive, gamepad,
radial ToF, matrix ToF). Firmware hash verified by esptool.

Stationary captures: `logs/diagnostics-2026-09-15/tof-before.jsonl` and
`tof-after.jsonl`, 563 raw samples each. Before: 385/563 samples were status 7
(WrapTargetFail); ranges were 57–75 mm. Old filtered output repeatedly restored
near returns, and 446/471 telemetry frames showed STOP. After: the physical fault
persisted (390/563 status 7, 58–74 mm), but every raw diagnostic result reported
filtered=-1, and all 470 telemetry frames showed CLEAR using the matrix's actual
front distances. Motor duty was zero in both captures. No drive command was sent.

The quality guard quarantines after eight failures in a 16-fresh-sample window,
recovers after 16 consecutive valid samples, and cannot recover on repeated
cached timestamps or gaps over 200 ms. Distance filtering remains unchanged for
healthy measurements. It reports missing coverage honestly, not a made-up far
range. Hardware/optical repair is still needed for reliable front-right radial
coverage; the matrix is not an identical replacement field of view.

Tests: 5 compiled quality-guard replays and 16 existing distance-filter tests
passed, including persistent real close obstacles, transient noise, time gaps,
clock rollover, invalid readings, and the live fault fixture.

## Follow-ups

Removed date-age expiration. Added durable `followup_asked_at`, separate from an
answered/completed event. Due events remain eligible until asked or resolved;
already-delivered follow-ups do not reappear after a restart even if unanswered.
Explicit rescheduling reopens the question for the new occurrence. Startup
follow-ups now record delivery via the speech queue's completed-playback callback,
not when the asynchronous task is submitted (which can later be dropped).

Applied the additive database column after an SQLite backup at
`logs/diagnostics-2026-09-15/people-before-followup-schema.db`. Historical event
flags were preserved: the old expiry flag is indistinguishable from some older
manual resolutions, and old records lack reliable question-delivery provenance.
This change does not automatically reopen previously expired records.

106 follow-up / speech / open-thread checks passed across seven isolated modules:
followup_asked (4), event_postponement (3), followup_resolution (12),
startup_followup_resolution (5), lean_event_followup (23), open_threads (49),
speech_engine (10). Compilation and diff whitespace checks passed.

## Voiceprint investigation

Active CAM++ rows: Bret #58; T'Joy #59 and #62; PJ #61. T'Joy's prints have
cosine similarity 0.541 to each other, versus 0.349 and 0.291 to Bret's saved
print. This comparison is evidence of separation, not proof of identity.
There are no archived original WAV recordings for these old enrollments.

#59 was saved September 7 at 19:02:00 on "My name is Joy, and I'm playing
Jeopardy." #62 was saved at 20:59:59 on "My name is T and I'm playing Jeopardy,"
originally under person 10 and later associated with person 3. That enrollment
used the old game-roster path despite an ambiguous resolver verdict and a bearing
estimate toward Bret. The owner confirmed during this investigation that T'Joy
herself said the second line. No voiceprints were deleted or reassigned.

No evidence here establishes that Bret's voice was saved as T'Joy. The September
15 false recognition plus authoritative-resolver override remains the demonstrated
identity failure; this task did not alter identity arbitration.
