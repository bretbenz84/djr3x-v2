# September 15, 21:12: verbal confirmation after a refused right turn

The run's initial come-here travelled 0.40 m, then held because independent
front-left range fell to about 1.45 m while camera range still said 4.07 m.
The alternate route issued left 15 degrees and forward 0.9144 m; its final
right 15 degree turn was refused. The separate right-turn request at 21:13:39
was also correctly parsed as -15 degrees (right/clockwise).

The turn refusal named the front-left arm, not a leftward command. The swing
model projected the fused front-left reading of 195 mm onto a radial bearing;
the independent front-left radial reading was 4000 mm. Rechecking the logged
right turn with that independent reading permits the full -15 degrees, while
retaining all other readings. The quarantined right radial remains unavailable.
The raw matrix grid would be needed to diagnose what caused the close return.

## Behavior

After a user-requested turn is refused by the predictive swing check, “yes you
can” (including “Yes, you can.”) takes a deterministic local confirmation path.
It consumes the pending turn once, preserving its signed angle, with a maximum
rate of 15 degrees/second. The confirmed attempt uses valid independent front
radial readings in place of fused front readings for the swing calculation.
Unavailable independent readings do not erase the corresponding fused reading.
Other sides, rear clearance, sensor freshness/health, charging, manual control,
and firmware stops remain enforced. No automatic escape translation or later
compass correction is attached to this confirmed attempt.

This is an override of the disputed matrix-based turn inference, not a blanket
disable of safety checks or a promise to execute through independently observed
obstacles. It applies to swing-refused turns; it does not override a come-here
personal-distance stop or a disconnected/dead-sensor refusal.

A pending turn expires after 45 seconds; a new issued/refused motion or explicit
stop/estop clears it. Sequence suppression cleanup preserves the refused step,
but cancels the route remainder. Confirmation retries only that step and never
replays earlier completed route legs. Weak voice identification does not prevent
the confirmation phrase from reaching the deterministic handler.

## Validation

126 isolated tests pass across `verbal_motion_confirmation` (7), `motion_swing`
(18), `motion_sequence` (4), `motion_route_tool` (79), and
`come_here_regression` (18), via `tools/run_lean_checks.py`.
Tests include the recorded rightward angle, one-shot consumption, expiry,
independent obstacles, missing independent range, stale telemetry/charging,
stop versus sequence cleanup, and the actual local speech handler with unknown
speaker identity. No live motor command or firmware change was made. The code
loads on the next Rex launch.
