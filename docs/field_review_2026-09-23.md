# Runtime review: 2026-09-23 20:53

Sources: `logs/djr3x-2026-09-23-20-53-03.log` and its matching
`conversation-2026-09-23-20-53-03.log`. Owner clarified the actual introduction
was “I'd like you to meet JT,” referring to Jeremy; ASR recorded J.C.

## Software fixes

- Several turns had ambiguous authoritative attribution but a named addressee
  in the reply agenda. For example, at 21:05:11 the best voice candidate was Bret
  (0.470 versus 0.417), yet visible Jeremy became the addressee. Reactive prompt
  construction now explicitly preserves an unidentified speaker in both the
  social frame and classic cast context. Visible faces remain scene context,
  not evidence that they spoke. Recognition thresholds are unchanged.
- At 21:06:26 Rex asked whether Jeremy was speaking; the owner corrected this
  to Bret. The preceding sample ranked Bret at 0.474 versus 0.240, while Jeremy
  was visible and had no active voiceprints. Enrollment now suppresses a
  speculative question to a visible person when a different enrolled voice is
  plausible with sufficient separation. It also cancels an unconfirmed pending
  enrollment for that other person. This suppression neither identifies the
  speaker nor authorizes learning. Jeremy still needs voice enrollment.
- “I'd like you to meet J.C.” at 21:00:00 fell through to ordinary chat. The
  introduction regex stopped at the first period and rejected single-letter J.
  Dotted initials now use the existing name normalizer before parsing, producing
  JC. The correctly transcribed “meet JT” also parses successfully. This fixes
  parsing, not the JT-to-J.C. transcription error, and does not automatically
  associate either nickname with Jeremy's record.
- The 20:56:52–53 repeated reply was logged on both sides of a motion notice.
  Streaming and the outer turn handler both wrote the completed reply, relying
  on adjacent-line deduplication. Streaming now owns the transcript entry and
  marks it as already logged for the outer handler. The log does not establish
  that audio played twice.

## Other findings and existing mitigations

- At 20:54:34 front-right radial ToF sensor 5 was quarantined for unreliable
  range status and reported -1. This is a known issue with existing mitigation,
  not a newly discovered defect: firmware filters and quarantines that channel,
  and front-pair fusion uses the matrix reading when the radial is invalid.
  The warning alone does not establish that this sensor blocked the later
  approach. No additional sensor repair or code change is proposed here.
- Turns repeatedly overshot: +45° reported +61.7°, -45° reported -62.7°, and
  +90° reported +111.1°. Calibration needs live checks of heading, stopping and
  actual rotation. No speed/braking/compass changes were inferred from this log.
- Garbled speech and a proactive forest/mural interpretation also occurred;
  the transcript alone does not establish a safe general correction.

## Validation

264 tests passed across 13 isolated modules using `tools/run_lean_checks.py`:
`field_2026_09_23`, `voice_learning`, `introductions_newcomer`,
`intro_misread_guards`, `conversation_streaming`, `lean_multi_party`,
`delivery_contract`, `two_chunk_tts`, `person_reference_identity`,
`group_room_behavior`, `conversation_revamp`, `conversational_persona`,
`voice_primary_identity`. Hardware/network access is disabled by that runner.
Physical speaker recognition and movement have not been retested.
