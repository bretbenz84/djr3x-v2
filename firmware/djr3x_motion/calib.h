// calib.h — measured drive-base constants + control gains (Phase 1).
//
// ⚠ THE GEOMETRY VALUES BELOW ARE PLACEHOLDERS, NOT MEASUREMENTS. ⚠
// They make odometry self-consistent but NOT accurate. Distances/angles will be
// wrong until you measure them on the real base (docs/motion_system.md §14):
//   1. Spin one wheel output-rev by hand; confirm COUNTS_PER_REV_OUTPUT.
//   2. Drive a measured 1.0 m straight; scale COUNTS_PER_METER by the error.
//   3. Spin a measured 360°; scale TRACK_WIDTH_M by the error.
// None of these affect safety (motion still needs an explicit command and the
// caps/watchdog/estop are independent) — only how far/how accurately it moves.
//
// The geometry values + the PID gains below are the BOOT DEFAULTS: they seed
// MotionParams and are runtime-overridable via the `config` command
// (counts_per_meter, track_width_m, kp/ki/kd), so you can calibrate + tune live with
// firmware/tools/motion_bench.py WITHOUT reflashing each iteration. Bake the winning
// values back here (or push them from config.py / .env) once you're happy.
#pragma once
#include <math.h>

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

// ---- Drive geometry (MEASURE — see above) --------------------------------
// Encoder spec (11 cycles/motor-rev/channel × 176:1 gear × 4 full-quad ≈ 7744
// counts/wheel-rev) is kept for reference but PROVED ~4.0× high on the bench —
// see COUNTS_PER_METER below. Either the datasheet PPR already included the
// quadrature factor or the gearing differs; the empirical value wins regardless.
#define COUNTS_PER_REV_OUTPUT  7744.0f   // reference only — superseded by empirical cpm
#define WHEEL_DIAMETER_M       0.080f    // reference only — superseded by empirical cpm
// EMPIRICAL turn calibration (bench `turn --deg 360` on hardwood, 2026-07-11,
// CONVERGED): 0.200 → 270° obs; 0.272 → ~330° obs; 0.297 → 360° observed = done.
// Effective track (includes contact-patch width in a spin) — expect a hair of
// over-rotation on carpet-grade scrub surfaces until the IMU (roadmap Phase A)
// makes heading surface-independent.
#define TRACK_WIDTH_M          0.297f

#define WHEEL_CIRCUM_M   ((float)M_PI * WHEEL_DIAMETER_M)
// EMPIRICAL distance calibration (bench `straight`, 2026-07-11, two passes):
// pass 1 — cmd 1.0 m, wall-blocked at odom 0.532 m, tape ~2.134 m (7 ft): the old
// derived value (COUNTS_PER_REV_OUTPUT/WHEEL_CIRCUM_M = 30812.39) was 4.01× high —
// every commanded speed had been ~4× fast physically. Provisional cpm 7683.
// pass 2 — cmd 1.0 m COMPLETED (odometry = exactly 1.00 by the completion rule),
// tape 1.02 m → 7683 × 1.00/1.02 = 7532. Residual now ~tape precision.
#define COUNTS_PER_METER 7532.0f

// Per-wheel count direction. +1 means "driving the wheel forward makes its count
// increase." Flip to -1 (per wheel) if the bench hand-turn test shows the sign
// backwards, instead of rewiring A/B.
#define ENC_SIGN_L  (+1.0f)
#define ENC_SIGN_R  (-1.0f)   // flipped: right side is mirror-mounted (bench `wheel right`
                              // 2026-07-11: forward cmd read +counts but spun BACKWARD —
                              // motor + encoder BOTH inverted; flip both, in lockstep)

// Per-wheel MOTOR direction. +1 means "a positive (forward) duty spins the wheel
// forward." Flip to -1 (per wheel) — the software equivalent of swapping that motor's
// two power leads — if the bench `spin` test runs away / auto-estops or the wheel spins
// BACKWARD on a forward command. MUST agree with ENC_SIGN_* (forward duty -> forward
// travel -> +counts): a mismatch makes the velocity PID positive feedback and trips the
// runaway guard. Fixing direction HERE keeps each channel paired with its own encoder,
// so it does NOT desync odometry the way swapping only the motor leads does.
#define MOTOR_SIGN_L  (+1)
#define MOTOR_SIGN_R  (-1)    // flipped in lockstep with ENC_SIGN_R (see above)

// ---- Motor PWM (LEDC) -----------------------------------------------------
// 20 kHz is above audible and well within the BTS7960's switching range.
#define PWM_FREQ_HZ    20000
#define PWM_RES_BITS   10                 // 0..1023 duty
#define PWM_DUTY_MAX   ((1 << PWM_RES_BITS) - 1)

// ---- Per-wheel velocity PID (target m/s → duty) ---------------------------
// ⚠ Starting gains — tune on the bench (docs §14.3): raise KP until the wheel
// tracks a step without buzzing, add KI to kill steady-state error, keep KD ~0
// unless it's oscillating. Units: duty per (m/s) of error.
// RESCALED ÷4.09 on 2026-07-11 when the cpm calibration made target/measured
// speeds REAL (they had been 4.09× inflated): the old gains were duty per
// INFLATED m/s, so keeping them would have quadrupled the loop gain overnight
// (overshoot/oscillation). Anchor: at the old FULL, kff put ~455 duty on the
// wheels for a true 0.716 m/s → plant ≈ 635 duty per real m/s.
#define WHEEL_PID_KP      440.0f
#define WHEEL_PID_KI      220.0f
#define WHEEL_PID_KD        0.0f
// Anti-windup: the integral's duty contribution is clamped to this so it can't
// accumulate while the motor is saturated/stalled.
#define WHEEL_PID_I_CLAMP  (0.8f * PWM_DUTY_MAX)

// ---- Velocity feedforward + stiction compensation -------------------------
// The PID alone starts every move from ZERO duty and only reaches a useful duty
// once the integrator winds up — so low speeds sit below breakaway friction (weak
// + slow to start) and duty scales with speed (strong only when fast). Two terms
// fix that mechanically:
//   KFF  — feedforward duty per commanded m/s. The instant a speed is commanded the
//          wheel gets ~the right duty, so the loop only trims instead of building
//          from nothing. Start ≈ 0.9 * PWM_DUTY_MAX / max_lin (so the top commanded
//          speed maps to ~90% duty, leaving PID headroom). Tune on the bench.
//   MIN_DUTY — a fixed breakaway "kick" added in the travel direction whenever a
//          nonzero speed is commanded, to clear static friction on a heavy base.
//          Raise until the wheel starts moving crisply at creep; lower if it lurches.
#define WHEEL_PID_KFF     640.0f    // duty per REAL m/s of COMMAND (was 2600 in the old
                                    // 4.09×-inflated units; 640 ≈ the measured plant gain)
#define WHEEL_MIN_DUTY    120.0f    // stiction breakaway kick (duty), in travel dir — duty
                                    // units, physical, NOT rescaled
// PIVOT (spin-in-place) breakaway: when the two wheel targets have OPPOSITE signs the
// base is scrubbing both tires sideways — a far higher static threshold than rolling,
// and higher again on carpet. STALL-GATED: the kick applies per wheel ONLY while that
// wheel is measurably stalled (|vmeas| < WHEEL_STALLED_EPS_MS) and drops back to
// WHEEL_MIN_DUTY within one 100 Hz tick of it rolling — a constant kick this large
// would massively overspeed the spin after breakaway (equilibrium ~5 rad/s vs a
// ~1.5 rad/s target), while stall-gating turns it into breakaway torque pulses.
// Field ladder (2026-07-11): the old "turning worked" hardwood baseline was ~780
// stalled duty (the inflated-units loop delivered it unknowingly); this carpet
// SATURATES at 1023 duty with ~zero rotation — in-place spins on that carpet exceed
// the platform's torque ceiling, full stop; use moving arcs there instead.
#define WHEEL_SPIN_BREAKAWAY_DUTY 700.0f
#define WHEEL_STALLED_EPS_MS      0.03f   // wheel counts as stalled below this (m/s);
                                          // ~2.3 encoder counts per 10 ms tick — resolvable

// A wheel target below this (m/s magnitude) counts as "stopped" → the wheel is
// braked to zero and its integrator reset rather than chasing micro-setpoints.
#define WHEEL_STOP_EPS_MS  0.01f

// ---- Drive setpoint slew (teleop feel) ------------------------------------
// Acceleration limit applied to the TELEOP (gamepad drive) setpoint so the base
// ramps smoothly toward the stick command in BOTH directions — symmetric, so a
// released stick coasts to a stop over ~(speed/accel) seconds instead of slamming
// to zero and dynamic-braking (the abrupt-stop complaint). Autonomous finite
// move/turn/come commands are NOT slewed here (they stay crisp + distance-accurate).
// Softened repeatedly after field tests ("takes off too fast" x3), then RESCALED
// 2026-07-11 when units became real. The old odometry units DEFLATED motion (lin
// ÷4.09, ang ÷2.75), so the field-approved feel in PHYSICAL units was: lin 0.2×4.09
// ≈ 0.8 m/s², ang 4.0×2.75 ≈ 11 rad/s². lin carries over exactly; ang is set a bit
// under the old physical value (8 < 11) — it was likely friction-limited anyway.
// (First pass wrongly set ang to 1.5 — a backwards conversion — and spins ramped
// like molasses, part of the "left/right has no power" report.)
// Tune with `set --accel-lin` / `--accel-ang` (higher = snappier).
#define DRIVE_ACCEL_LIN    0.8f     // m/s^2  (teleop linear setpoint slew, REAL units)
#define DRIVE_ACCEL_ANG    8.0f     // rad/s^2 (teleop angular setpoint slew, REAL units)
// (The old DRIVE_SPIN_LIN_EPS binary spin gate is gone — replaced by the smooth
// GAMEPAD_SPIN_BLEND_FWD_LO/HI band above; the blend factor rides the setpoint.)

// ---- ToF subsystem (8 radial sensors) — only used when MOTION_TOF_PRESENT==1 -
// Mounted at the 540 mm base-ring surface, every 45° starting 22.5° off the forward
// axis: 4× short-range VL53L0X on mux ch 0-3 (LEFT pair lf/lb + RIGHT pair rf/rb) +
// 4× long-range VL53L1X on mux ch 4-7 (FRONT pair fl/fr + REAR pair rl/rr).
// Requires the TCA9548A mux (8 sensors > free XSHUT GPIOs). docs §6.
#define TOF_SHORT_COUNT       4         // VL53L0X (short), mux ch 0..3 — left/right pairs
#define TOF_LONG_COUNT        4         // VL53L1X (long),  mux ch 4..7 — front/rear pairs
#define TOF_COUNT             (TOF_SHORT_COUNT + TOF_LONG_COUNT)   // 8 total
#define TOF_MUX_ADDR          0x70      // TCA9548A I²C address (mux selects one ch at a time)
// Per-read wait for a fresh continuous sample. MUST exceed the slowest sensor's
// inter-measurement period (L1X, below) or a live-but-slow sensor reads as -1 while
// we wait for its next sample. Dead sensors are skipped (s_ok), so this only bounds
// a genuinely stuck live sensor.
#define TOF_TIMEOUT_MS        100
// A sensor whose reads keep failing must not FREEZE its last-good distance forever —
// a sensor that dies while reading "clear" would silently disable the stop reflex in
// its direction (an undetectable safety hole). After this many CONSECUTIVE failed
// reads (each sensor is revisited every ~80 ms, so 8 ≈ 0.6 s) the published distance
// drops to -1: an honest error the GUI radar/telemetry can show. safety.cpp still
// fails OPEN on -1 by documented choice (min2_valid skips it; the pair partner covers).
#define TOF_ERR_STREAK_STALE  8

// Fast-attack / slow-release output filter (field fix 2026-07-11): the published
// distance drops to a NEARER reading INSTANTLY (danger is never filtered) but rises
// toward a farther one at most this many mm per revisit (~80 ms/sensor). The long
// sensors sit 45° apart pointing outward, so a narrow obstacle (chair leg) at a
// beam's edge strobed 0.5 m <-> 4 m as it entered/left the cone — flapping the GUI
// radar, the steering assist, AND the stop reflex (BLOCKED released on each "clear"
// frame and the base lurched forward). With the filter the close return HOLDS and
// decays smoothly: ~300 mm/poll ≈ 3.7 m/s of release, a cleared wall reads fully
// open again in ~1 s.
#define TOF_RELEASE_STEP_MM       300

// VL53L0X (short range, ~1.2 m reliable):
#define TOF_L0X_TIMING_BUDGET_US  33000 // 33 ms measurement budget (speed vs accuracy)
#define TOF_L0X_OUT_OF_RANGE_MM   2000  // clamp "nothing in range" to a far/clear value

// VL53L1X (long range): Long mode reaches ~4 m; needs a larger timing budget than L0X.
// The inter-measurement period must be >= the timing budget (+overhead), else the
// sensor won't produce readings (datasheet); 60 ms > 50 ms budget satisfies that.
#define TOF_L1X_TIMING_BUDGET_US  50000 // 50 ms budget (Long mode wants >= ~33 ms)
#define TOF_L1X_INTERMEASUREMENT_MS 60  // continuous-mode period (> timing budget)
#define TOF_L1X_OUT_OF_RANGE_MM   4000  // clamp "nothing in range" to a far/clear value

// ---- Hallway steering assist (manual forward drive) ------------------------
// While the gamepad commands FORWARD, the base steers itself away from walls using
// the side short-range pairs (lateral clearance) plus the front long pair
// (anticipatory: approaching a wall at an angle steers toward the open side). Sized
// for a typical US hallway (~915-1220 mm) around the 540 mm base ring: centered in a
// 1 m hall each side reads ~230 mm, well inside ENGAGE, so both walls are "felt" and
// the correction centers the base. Walls beyond ENGAGE are ignored (open rooms drive
// exactly as before). The operator's stick adds on top; the correction itself is
// capped at ASSIST_MAX_ANG_FRAC of max_ang so the human always has override
// authority; the Z_STOP reflex still hard-blocks a head-on wall regardless.
#define ASSIST_ENGAGE_MM      450.0f    // walls farther than this don't steer
#define ASSIST_GAIN           2.0f      // rad/s per METER of left-right imbalance
#define ASSIST_FRONT_WEIGHT   0.7f      // front-pair contribution vs the side pairs
#define ASSIST_MAX_ANG_FRAC   0.6f      // correction cap as a fraction of max_ang
#define ASSIST_MIN_LIN_MS     0.02f     // assist only while actually driving forward
// Close-wall REPULSION (field fix 2026-07-11: base scraped hallway walls the pure
// imbalance term couldn't prevent — centered in a too-narrow gap the left/right
// difference reads ~zero, so no steer, and an equally-close pair got NO correction
// at all). A side wall inside REPEL (~5 in) now pushes back hard on its own,
// proportional to penetration, independent of the other side; both close -> net
// push away from the NEARER wall. Rides the same ASSIST_MAX_ANG_FRAC cap, so the
// operator keeps override authority.
#define ASSIST_REPEL_MM       130.0f    // ~5 in: a side wall inside this repels hard
#define ASSIST_REPEL_GAIN     12.0f     // rad/s per METER of penetration inside REPEL

// ---- Battery sense (INA226, battery.cpp) -----------------------------------
// Shunt resistance in MICRO-ohms for the current (batt_ma) reading; 0 disables
// current and reports voltage only. AS BUILT (2026-07-11): an external R002
// (2 mΩ, 20 A) shunt inline in the aggregate battery NEGATIVE lead — pack− →
// shunt → fuse block → devices — with the module's IN+/IN− sensed across it
// (low-side) and VBUS at the fuse block's positive. 2 mΩ across the INA226's
// ±81.9 mV input = ±41 A range at ~1.25 mA/LSB. (First bring-up ran with the
// stock-module R100 value 100000 here — every reading was ÷50.)
#define BATT_SHUNT_MICROOHM  2000
// Sense polarity: +batt_ma = DISCHARGING (protocol §6.1). The as-built IN+/IN−
// orientation reads discharge negative, so flip in software rather than rewire.
#define BATT_CURRENT_SIGN    (-1)

// ---- Speed-adaptive zone envelope (safety.cpp zones + control.cpp taper) ----
// The front/rear pairs are angled ±22.5° off the travel axis, so at RANGE they see
// obstacles outside the actual collision corridor (at 0.9 m a beam points ~0.35 m
// off-centerline — wider than the 0.27 m half-body), while everything they see CLOSE
// is genuinely in the path. Fixed zones therefore over-brake at distance and
// under-serve precision parking. The effective zones now SCALE WITH MEASURED SPEED
// (context.h stop_zone_eff/slow_zone_eff): the configured slow_zone_m/stop_zone_m
// are the FULL-SPEED envelope (braking distance matters, angled false positives
// cost little), shrinking linearly to the floors below at rest — where the operator
// is deliberately positioning and the angled beams only see true in-path obstacles.
// The STOP floor is the "never able to actually hit the wall" guarantee.
#define STOP_ZONE_MIN_M       0.10f   // hard-stop floor at rest (~4 in) — never hittable
#define SLOW_ZONE_MIN_M       0.18f   // braking-band floor at rest
#define ZONE_SPEED_REF_MS     0.60f   // speed at which the configured zones fully apply
                                      // (was 0.18 in the old inflated units; full teleop
                                      // is now a REAL ~0.72 m/s)

// ---- Approach slowdown creep floor (control.cpp slow-zone taper) ------------
// The progressive slow-zone taper scales the commanded speed toward zero at the stop
// line — but the loaded base needs a REAL speed to move at all (stiction; the same
// reason GAMEPAD_SPEED_SLOW had to be raised at full build weight), and the wheel PID
// treats targets under WHEEL_STOP_EPS_MS as a stop. Without a floor the taper stalls
// the base mid-slow-zone, far from the wall (field-logged 2026-07-11: "won't let me
// within 3 feet"). Inside the slow zone the command is floored at this creep speed
// (never MORE than the operator commanded), so the base crawls under control all the
// way to the stop_zone hard block instead of dying early.
#define APPROACH_CREEP_MIN_MS 0.06f     // m/s; comfortably above WHEEL_STOP_EPS_MS + stiction

// ---- Bluetooth gamepad (Bluepad32) — only when MOTION_GAMEPAD_PRESENT==1 ----
// Left stick = arcade drive (Y forward, X turn); L1 creep / R1 boost; B = e-stop;
// Start = clear + return to AUTO; hold BOTH analog triggers = full-override (docs §11).
#define GAMEPAD_DEADZONE       0.12f    // stick fraction ignored around center
// TELEOP CEILINGS — the gamepad's own speed caps, in REAL m/s / rad/s, independent
// of params.max_lin/max_ang (2026-07-11): the params caps are the AUTONOMOUS limits
// and the Mac pushes them DOWN on connect (config.py MOTION_MAX_LINEAR_MS = 0.25),
// which used to silently cap teleop too — the pad crawled whenever Rex was running.
// control_tick clamps a MANUAL drive to these (bounded by the hard caps) and
// autonomous motion to the params caps, so the two are finally decoupled.
// 0.72 m/s = the physical top speed the base was ACTUALLY field-driven at daily
// under the old 4.09× cpm miscalibration (the approved feel, now in honest units).
#define GAMEPAD_MAX_LIN_MS     0.72f    // teleop linear ceiling (level FULL = 1.00 × this)
#define GAMEPAD_MAX_ANG_RADS   2.20f    // teleop turn ceiling. Field fix (same day): first
                                        // set to 0.80 from a BACKWARDS unit conversion and
                                        // spins lost their breakaway torque ("left/right has
                                        // no power") — the old "1.5" ceiling was in odometry
                                        // units that DEFLATED rotation 2.75×, i.e. physically
                                        // ~4.1 rad/s of authority and ~510 duty of spin kick.
                                        // 2.20 (under the 2.5 hard cap) restores ~475 duty of
                                        // kick; full-stick spin ≈ 126°/s. Tune by feel.
// Teleop speed levels, cycled by CLICKING the left stick (L3): slow -> faster -> full ->
// slow. Boots at SLOW so the default is gentle; each is a fraction of GAMEPAD_MAX_LIN_MS.
// History: repeatedly retuned in the old inflated units (bare-base / full-weight /
// carpet passes); when units became real (2026-07-11) the fractions were recomputed so
// each level's PHYSICAL speed matches what those field passes actually approved:
// SLOW 0.43, MED 0.60, FULL 0.72 m/s.
#define GAMEPAD_SPEED_SLOW     0.60f    // level 0 (default on boot / reconnect) ≈ 0.43 m/s
#define GAMEPAD_SPEED_MED      0.83f    // level 1 ≈ 0.60 m/s
#define GAMEPAD_SPEED_FULL     1.00f    // level 2 = 0.72 m/s (the approved top speed)
// Forward/back stick RESPONSE CURVE: lin command = sign(fwd)*|fwd|^GAMMA * level max.
// GAMMA < 1 is concave ("anti-expo"): more authority at small stick pushes — at 25%
// stick you command ~|0.25|^0.6 ≈ 44% of the level's max (linear gave 25%) — while full
// deflection still hits exactly the level max, so top speeds are unchanged. This is what
// makes the loaded base actually MOVE at small/medium stick without a linear speed-up.
// 1.0 = linear (old feel). Applies to the LINEAR axis only; the turn axis stays linear
// and the spin↔arcade blend keys off the RAW stick so the tuned blend bands don't shift.
#define GAMEPAD_LIN_GAMMA      0.60f
// A pure in-place SPIN (stick full left/right, no forward/back) uses THIS turn scale
// instead of the speed level, so ALL levels get the same full turning authority — enough
// feedforward duty to break carpet traction and actually rotate. The speed level throttles
// TRANSLATION, not the pivot. 1.0 = the full max_ang cap (tune the spin rate with `set
// --max-ang`; the PID saturates to max duty on a stiff surface regardless).
#define GAMEPAD_SPIN_SCALE     1.00f
// Spin↔arcade BLEND band, on the forward/back stick fraction (post-deadzone, 0..1).
// Below LO: pure spin-in-place (full authority, inside wheel may reverse). Above HI:
// pure arcade steer (level-scaled authority, inside wheel floored at 0). Between the
// two, BOTH the turn authority and the wheel mixing interpolate smoothly (smoothstep),
// so tilting slightly forward out of a spin eases into a tightening arc instead of the
// turn rate collapsing at a hard threshold (field bug: "mostly left + slightly forward
// acts strange" — the old binary 0.02 m/s gate stepped authority 1.0 -> 0.15 at slow).
#define GAMEPAD_SPIN_BLEND_FWD_LO  0.05f   // stick fraction where the blend starts
#define GAMEPAD_SPIN_BLEND_FWD_HI  0.35f   // stick fraction where it's fully arcade
// D-pad driving nudges (rising edge, one per press): Up/Down = a short finite forward/
// back MOVE (encoder-closed, ToF stop-reflex gated like any move); Left/Right = a
// RELATIVE turn by params.default_turn_deg (90°). Hold L1 + D-pad for the original
// absolute-heading encoder test instead (bring-up/calibration only).
#define GAMEPAD_NUDGE_DIST_M      0.30f  // Up/Down nudge travel (m)
#define GAMEPAD_NUDGE_SPEED_FRAC  0.30f  // nudge speed as a fraction of max_lin
#define GAMEPAD_TRIGGER_MAX    1023.0f  // Bluepad32 analog trigger full-scale
#define GAMEPAD_FULL_OVERRIDE_FRAC 0.85f // both triggers past this fraction = bypass ToF
#define GAMEPAD_TRIGGER_PRESS_FRAC 0.50f // trigger past this fraction = "pressed" (GUI mirror)
// Rumble (force feedback, playDualRumble): tactile echo of the collision avoidance +
// a hello greeting when a host (main.py) connects. All Bluepad32 calls stay on the
// Arduino loopTask (gamepad_tick); other tasks only set pending flags.
#define GAMEPAD_RUMBLE_ENABLED        1     // build-time master switch (0 = never rumble)
#define GAMEPAD_RUMBLE_BLOCK_MS       320   // hard-stop (BLOCKED) thump duration
#define GAMEPAD_RUMBLE_BLOCK_STRONG   0xE0  // heavy motor: a real thump
#define GAMEPAD_RUMBLE_BLOCK_WEAK     0x30
#define GAMEPAD_RUMBLE_BLOCK_REPEAT_MS 700  // re-thump cadence while still pushing into it
#define GAMEPAD_RUMBLE_SLOW_MS        130   // entering the braking band: light buzz
#define GAMEPAD_RUMBLE_SLOW_WEAK      0x80  // light motor only — clearly distinct from the thump
#define GAMEPAD_RUMBLE_SLOW_STRONG    0x00
#define GAMEPAD_RUMBLE_HELLO_MS       150   // host-connect greeting: friendly double pulse
#define GAMEPAD_RUMBLE_HELLO_GAP_MS   180
#define GAMEPAD_RUMBLE_HELLO_MAG      0x70
#define GAMEPAD_RUMBLE_HELLO_TTL_MS   3000  // drop a stale greet if no pad is on within this
