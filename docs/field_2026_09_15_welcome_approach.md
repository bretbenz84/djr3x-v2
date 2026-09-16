# Restoring the silent first-face welcome approach

The 19:48 run launched its startup approach at 19:48:46 with fused front
clearance 1.82 m, passing the old 1.8 m startup threshold. It travelled 2.59 m
before the old emergency-only front guard stopped it. Later approach repairs
introduced stand-off braking and separated broad matrix obstacle range from
independent radial range, but the startup eligibility gate still demanded
1.8 m from the fused minimum. Readings such as the later logged `fr=793`,
`fl_radial=4000`, `fr_radial=-1` could pass the repaired approach controller
but never launch a welcome approach. The old gate did not log its reason.

The welcome gate now uses the shared approach clearance budget. It requires
fresh telemetry, at least 1.8 m of independent front range (conservative fused
fallback when independent range is unavailable), and at least 15 cm of usable
clearance budget. Matrix obstacle protection remains active. A changed target
or missing face resets the two-tick confirmation. Clearance holds are logged.

The motion is a bounded welcome step: 0.10 m/s maximum configured speed, 60 cm
travel budget with advance braking, and 1.30 m requested personal stand-off.
Whichever constraint arrives first stops the approach. It runs through the
same camera-targeted controller as come-here, retaining camera freshness,
obstacle checks, sensor loss stops and firmware protection. It is not intended
to cross the whole room to a distant face.

No spoken greeting, voice identity, or spoken invitation is required. Existing
face tracking selects the target; it can be an unrecognized face. The 180-second
opportunity starts at first tracked face rather than spending itself while the
room is empty. Only one welcome is launched per run. Explicit motion requests,
stay-put/room rules, games, mid-sentence holds, charging, traction problems and
critical battery retain priority. A failed/blocked launched welcome is not
automatically retried.

Validation: 295 hardware-isolated tests pass in `motion_agency` (230),
`come_arrival` (20), `come_target_acquisition` (27), and `come_here_regression`
(18). Added cases cover the recorded matrix disagreement, actual close
obstacles, stale telemetry, face disappearance, no voice identity, late first
face after an empty room, and braking within the welcome travel budget with
a badly overestimated camera range. No live drive or firmware change was made;
these Mac-side changes load on Rex's next launch. Sensor calibration and the
faulty right radial remain physical limitations.
