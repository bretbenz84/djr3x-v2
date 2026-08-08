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
// counts/wheel-rev) is kept for reference but proved substantially high on the
// bench — see COUNTS_PER_METER below. Either the datasheet PPR already included
// the quadrature factor or the gearing differs; the empirical value wins regardless.
#define COUNTS_PER_REV_OUTPUT  7744.0f   // reference only — superseded by empirical cpm
#define WHEEL_DIAMETER_M       0.100f    // measured rubber drive-wheel diameter
// EMPIRICAL turn calibration. Old mixed-motor build (2026-07-11): 0.200 → 270°;
// 0.272 → ~330°; 0.297 → 360°. Current dual-60RPM build field run (2026-07-21):
// cmd 180° produced ~270° at 0.297, so standard proportional correction gives
// 0.297 * 270/180 = 0.4455 m. Effective track includes contact-patch scrub and any
// residual average encoder-scale error; refine with a taped 360° test on hardwood.
#define TRACK_WIDTH_M          0.446f

#define WHEEL_CIRCUM_M   ((float)M_PI * WHEEL_DIAMETER_M)
// EMPIRICAL distance calibration (bench `straight`, 2026-07-11, two passes).
// The original calculation mistakenly assumed an 80 mm wheel; the fitted wheels
// are 100 mm. The tape-measured counts-per-metre below remains authoritative:
// pass 1 — cmd 1.0 m, wall-blocked at odom 0.532 m, tape ~2.134 m (7 ft): the old
// old 80 mm derived value (30812.39 counts/m) was 4.01× high —
// every commanded speed had been ~4× fast physically. Provisional cpm 7683.
// pass 2 — cmd 1.0 m COMPLETED (odometry = exactly 1.00 by the completion rule),
// tape 1.02 m → 7683 × 1.00/1.02 = 7532. Residual now ~tape precision.
// SPLIT PER WHEEL 2026-07-17: the replacement 60RPM left motor has a different
// gearbox ratio, so its encoder scale differs from the right's. 7532 is the
// EMPIRICAL right-wheel value (tape-calibrated 2026-07-11). The left value is
// measured with the hand-roll test (bench `encoder`: one full wheel rev =
// pi*0.100 m of travel). Runtime keys: counts_per_meter_l / _r (the legacy
// counts_per_meter config key sets BOTH, for the bench straight calibration).
#define COUNTS_PER_METER_L 17000.0f  // MEASURED 2026-07-17 (hand-roll, one full rev =
                                     // 5341 counts): the 60RPM gearbox is 2.26x the
                                     // right's ratio. Re-measure when the matching
                                     // right motor arrives.
#define COUNTS_PER_METER_R 17000.0f  // ⚠ ASSUMES the matching 60RPM right motor is
                                     // FITTED (arriving 2026-07-20) — seeded from the
                                     // left's measured value; VERIFY with the hand-roll
                                     // before trusting (gearbox tolerance varies a few
                                     // % even within a SKU). History: the old right
                                     // measured 3361, and the old shared 7532 was the
                                     // average of two secretly-mismatched gearboxes —
                                     // the per-wheel split exposed it (and the true
                                     // culprit behind 2026-06-23 "left motor always
                                     // slow"). DO NOT flash this onto the old motor.

// Per-wheel count direction. +1 means "driving the wheel forward makes its count
// increase." Flip to -1 (per wheel) if the bench hand-turn test shows the sign
// backwards, instead of rewiring A/B.
#define ENC_SIGN_L  (-1.0f)   // flipped 2026-07-17: replacement 60RPM left motor is
                              // mirror-wired vs the original (bench `wheel left`:
                              // + counts but physically BACKWARD — flip BOTH signs)
#define ENC_SIGN_R  (+1.0f)   // 2026-07-20: right motor replaced with the 60RPM SKU. The new
                              // unit reads/drives inverted vs the old right, so flip to +1
                              // (opposite the left's -1: identical motor mounted mirror-image).
                              // Encoder + motor flipped BOTH in lockstep; bench-verify `wheel right`.

// Per-wheel MOTOR direction. +1 means "a positive (forward) duty spins the wheel
// forward." Flip to -1 (per wheel) — the software equivalent of swapping that motor's
// two power leads — if the bench `spin` test runs away / auto-estops or the wheel spins
// BACKWARD on a forward command. MUST agree with ENC_SIGN_* (forward duty -> forward
// travel -> +counts): a mismatch makes the velocity PID positive feedback and trips the
// runaway guard. Fixing direction HERE keeps each channel paired with its own encoder,
// so it does NOT desync odometry the way swapping only the motor leads does.
#define MOTOR_SIGN_L  (-1)    // flipped in lockstep with ENC_SIGN_L (see above)
#define MOTOR_SIGN_R  (+1)    // 2026-07-20: flipped in lockstep with ENC_SIGN_R (60RPM right-motor
                              // swap). Bench-verify with `spin` / `wheel right` — forward duty must
                              // spin FORWARD and not trip the runaway guard.

// ---- Motor PWM (LEDC) -----------------------------------------------------
// 20 kHz is above audible and well within the BTS7960's switching range.
// (Weak-torque investigation 2026-07-12: a 5 kHz experiment produced IDENTICAL
// behavior — same ~1.8 A pack draw at dual near-full duty, same non-rotation —
// so PWM gate-drive loss is RULED OUT. The measured signature — full-duty motor
// current nearly invisible at the pack shunt, no voltage sag — points at the
// BTS7960 B+/B- supply not actually coming from the main pack path; hardware
// trace pending.)
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
// PER-WHEEL GAIN SCALES (2026-07-17, mixed motors): the shared gains below were
// tuned in the old shared-7532 encoder units. The per-wheel encoder split made
// measured speeds TRUE m/s, which changed each wheel's effective plant gain:
//   right (old motor, true cpm 3361): plant = 640 * 3361/7532 ≈ 286 duty/(m/s) → 0.446x
//   left (new 60RPM motor): ~1023 duty ≈ 0.31 m/s → plant ≈ 3300 duty/(m/s) → ~5.2x
// Applied to kp/ki/kff (not min_duty — breakaway duty is a motor property, not a
// unit artifact). Runtime keys: gain_scale_l / gain_scale_r. Re-derive when the
// matching 60RPM right motor arrives (both will be ~5.2x, i.e. retune the bases).
#define WHEEL_GAIN_SCALE_L  5.2f
#define WHEEL_GAIN_SCALE_R  5.2f     // ⚠ matching-60RPM value (old right motor was
                                     // 0.446) — flash only after the motor swap; then
                                     // fine-tune BOTH on the floor as a pair
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
#define WHEEL_MIN_DUTY    230.0f    // running duty floor (duty), in travel dir — applies
                                    // while the wheel is ROLLING. MEASURED on the
                                    // full-weight base (2026-07-12): sustained creep
                                    // needs ~240 total duty; at the old 120 the wheel
                                    // re-stalled one tick after breakaway (100 Hz
                                    // stick-slip chatter, "only moves at full throttle")
// ---- Battery-voltage feedforward compensation -----------------------------
// Duty is a fraction of PACK voltage: applied_volts = (duty/PWM_DUTY_MAX)*V_batt.
// KFF/MIN_DUTY/breakaway above are all calibrated in DUTY at a healthy pack, so as
// a LiFePO4 pack sags under sustained drive current (and worse across the ~160mOhm
// junction) the SAME duty delivers fewer volts -> the wheels slow, and near the top
// of the range the ~90% feedforward has too little PID headroom left to claw it back
// (the "full-speed run audibly slows over time" report, 2026-07-20). Compensation
// scales the commanded duty by V_NOMINAL/V_batt so effective volts stay constant as
// the pack droops — bounded so a bad/absent sensor can only ever be a no-op or a
// modest, physically-clamped boost.
//   • batt_mv unknown (-1, no INA226) or outside [MIN,MAX]_VALID -> factor = 1.0 (off).
//   • Factor is clamped to [1.0, COMP_MAX]: only ever BOOSTS a sagging pack, never
//     trims a fresh one (keeps present full-charge top-speed feel unchanged).
//   • The boost is still hard-limited by PWM_DUTY_MAX in wheel_pid — 100% duty is the
//     physical ceiling; comp only recovers headroom BELOW it, it can't exceed the pack.
#define BATT_COMP_NOMINAL_MV   12800   // LiFePO4 4S nominal; the voltage KFF assumes
#define BATT_COMP_MAX          1.30f   // never boost duty more than +30%
#define BATT_COMP_MIN_VALID_MV 9000    // below this, treat the reading as a glitch -> off
#define BATT_COMP_MAX_VALID_MV 16000   // above this, treat the reading as a glitch -> off

// STRAIGHT-drive breakaway (owner 2026-07-12: the full-weight robot needs a
// substantial duty punch to leave a dead stop — below it, low commands just hummed).
// STALL-GATED like the pivot tiers: while a wheel is COMMANDED but measured
// stationary, its duty floors here (owner's 35% start was MEASURED marginal — twitch,
// not launch — on the full-weight base; ~49% breaks away decisively —
// tune live with `set --breakaway`); one 100 Hz tick after it rolls, the floor drops
// to WHEEL_MIN_DUTY and the closed loop holds the actual commanded speed. Net feel:
// dead stop -> immediate 35% punch -> rolling -> true low-speed control. Runtime
// param (breakaway_duty); this is the boot default.
#define WHEEL_STRAIGHT_BREAKAWAY_DUTY 500.0f
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
// Pivot RUNNING floor (field fix, same day): stall-gating alone made spins twitch —
// the kick broke the wheels loose, dropped to WHEEL_MIN_DUTY (120) one tick later,
// and the SUSTAINED sideways scrub of a pivot re-stalled them (stick-slip grind;
// the integrator needed seconds to build the missing duty). While pivoting and
// ROLLING, the kick floors here instead of at 120, carrying the scrub load; the
// per-surface profiles (GAMEPAD_*_SPIN_RUN) override this base at mode-apply.
#define WHEEL_SPIN_RUN_DUTY       380.0f
#define WHEEL_STALLED_EPS_MS      0.03f   // wheel counts as stalled below this (m/s);
                                          // ~2.3 encoder counts per 10 ms tick — resolvable

// A wheel target below this (m/s magnitude) counts as "stopped" → the wheel is
// braked to zero and its integrator reset rather than chasing micro-setpoints.
#define WHEEL_STOP_EPS_MS  0.01f

// Do not treat the first encoder twitch as proof that the full chassis has broken
// static friction. Gear lash and tyre compliance can produce several counts while
// the base is still planted. A launch must show sustained wheel motion before the
// controller drops from the breakaway tier to the running tier; likewise, a rolling
// wheel must remain stalled for a while before breakaway is re-armed.
#define WHEEL_LAUNCH_CONFIRM_S   0.12f
#define WHEEL_RESTALL_CONFIRM_S  0.12f

// Rated 176 rpm with a 100 mm wheel is 0.922 m/s unloaded. Keep mixed wheel-speed
// requests just below that physical ceiling and scale both sides together so a
// combined forward+turn command preserves curvature instead of saturating only the
// outside wheel.
#define WHEEL_TARGET_MAX_MS      0.90f

// ---- Drive setpoint slew (teleop feel) ------------------------------------
// Acceleration limit applied to every normal drive setpoint so the base
// ramps smoothly toward the stick command in BOTH directions — symmetric, so a
// released stick coasts to a stop over ~(speed/accel) seconds instead of slamming
// to zero and dynamic-braking. Autonomous finite move/turn/come share the ramp.
// Softened repeatedly after field tests ("takes off too fast" x3), then RESCALED
// 2026-07-11 when units became real. The old odometry units DEFLATED motion (lin
// ÷4.09, ang ÷2.75), so the field-approved feel in PHYSICAL units was: lin 0.2×4.09
// ≈ 0.8 m/s², ang 4.0×2.75 ≈ 11 rad/s². lin carries over exactly; ang is set a bit
// under the old physical value (8 < 11) — it was likely friction-limited anyway.
// (First pass wrongly set ang to 1.5 — a backwards conversion — and spins ramped
// like molasses, part of the "left/right has no power" report.)
// Tune with `set --accel-lin` / `--accel-ang` (higher = snappier).
#define DRIVE_ACCEL_LIN    1.2f     // m/s^2  (teleop linear setpoint slew, REAL units)
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

// Glitch confirmation (tof_filter.h, field data 2026-07-21): a reading that
// drops the published distance by more than ATTACK_DROP in one frame is only
// believed when the NEXT frame agrees within ATTACK_CONFIRM — kills the
// single-frame phantom near returns (VL53L1X wrap-around, VL53L7CX speckle)
// that sawtoothed fl/fr/rr between 1 m and 4 m in an empty room. Costs one
// sensor revisit (~75-80 ms) of detection latency on a genuinely sudden
// obstacle; smooth approaches (< DROP_MM per frame) still attack instantly.
#define TOF_ATTACK_DROP_MM        400
#define TOF_ATTACK_CONFIRM_MM     300

// VL53L0X (short range, ~1.2 m reliable):
#define TOF_L0X_TIMING_BUDGET_US  33000 // 33 ms measurement budget (speed vs accuracy)
#define TOF_L0X_OUT_OF_RANGE_MM   2000  // clamp "nothing in range" to a far/clear value

// VL53L1X (long range): Long mode reaches ~4 m; needs a larger timing budget than L0X.
// The inter-measurement period must be >= the timing budget (+overhead), else the
// sensor won't produce readings (datasheet); 60 ms > 50 ms budget satisfies that.
#define TOF_L1X_TIMING_BUDGET_US  50000 // 50 ms budget (Long mode wants >= ~33 ms)
#define TOF_L1X_INTERMEASUREMENT_MS 60  // continuous-mode period (> timing budget)
#define TOF_L1X_OUT_OF_RANGE_MM   4000  // clamp "nothing in range" to a far/clear value

// ---- 8x8 Matrix ToF (DFRobot SEN0628 — MOTION_TOF_MATRIX_PRESENT==1) --------
// Front-mounted, level, VL53L7CX behind an onboard RP2040 (tof_matrix.cpp).
// Range 20-3500 mm, 45°x45° FOV (63° diagonal), 8x8 = 15 Hz max on the sensor.
#define TOF_MATRIX_ADDR            0x33  // DIP-strapped I2C address (bench-confirmed)
#define TOF_MATRIX_FRAME_INTERVAL_MS 75  // ~13 Hz poll (sensor tops out at 15 Hz in 8x8)
#define TOF_MATRIX_READ_TIMEOUT_MS   30  // hard deadline for one frame read — a slow/
                                         // wedged RP2040 costs at most this per attempt
#define TOF_MATRIX_MODE_ACK_TIMEOUT_MS 8000 // SETMODE ack wait (init task only). The
                                         // RP2040 only acks AFTER the ~5 s VL53L7CX
                                         // reconfigure; a short window here re-sends
                                         // SETMODE into a busy sensor, which wedges
                                         // the I2C trunk and hangs the whole firmware
                                         // (field-observed 2026-07-16). Must exceed
                                         // the reconfigure time; vendor lib polls 8 s.
#define TOF_MATRIX_MODE_SETTLE_MS  5200  // VL53L7CX reconfigure settle (vendor lib uses 5000)
#define TOF_MATRIX_STALE_MS         500  // publisher silence past this = poll task dead
                                         // (I2C wedge) -> fl/fr report -1, robot stays alive
//
// FLOOR REJECTION geometry — the sensor sits above the floor, so the lower rows
// permanently see floor at short range (h=0.15 m: bottom row ≈ 445 mm along-ray).
// Per-row expected floor distance = HEIGHT / sin(row angle below horizon); readings
// at/beyond FLOOR_TOLERANCE of it are floor (clear), meaningfully shorter = obstacle.
// ⚠ MEASURE TOF_MATRIX_HEIGHT_M on the robot: lens centre to floor, metres. If unsure
// err HIGH — too-high reads the empty floor as an obstacle (nuisance block, obvious);
// too-low classifies real low obstacles as floor (missed, silent).
#define TOF_MATRIX_HEIGHT_M        0.11f // EFFECTIVE optical height — empirically
                                         // calibrated 2026-07-16. Tape-measured lens
                                         // height is 0.16 m, but the raw grid on open
                                         // floor reads row7≈330/row6≈460 mm (vs 474/657
                                         // predicted): oblique ToF returns under-range
                                         // at grazing angles by a consistent ~0.70x.
                                         // 0.11 makes both bottom rows match observed
                                         // floor. Recalibrate if the mount height moves.
#define TOF_MATRIX_PITCH_DEG       0.0f  // mount pitch trim (+ = tilted up); level = 0
#define TOF_MATRIX_VFOV_DEG        45.0f // VL53L7CX vertical FOV (45° square per ST)
#define TOF_MATRIX_FLOOR_TOLERANCE 0.80f // reading >= this fraction of expected floor = floor
#define TOF_MATRIX_FLOOR_MIN_MM    250   // never call anything closer than this "floor"
#define TOF_MATRIX_MIN_MM          25    // below sensor min range = speckle, ignore
#define TOF_MATRIX_CLEAR_MM        3500  // "nothing in range" clear value (sensor max)
//
// Orientation — normalize the raw grid so row 0 = physically TOP and column 0 = the
// ROBOT'S LEFT edge of the FOV. ⚠ VERIFY ON THE BENCH with tools/tof_matrix_gui.py:
// (1) tilt the module down — if the NEAR readings move to the TOP rows, set FLIP_V 1;
// (2) hold a hand at the robot's front-LEFT — if fr (not fl) drops, set FLIP_H 1.
#define TOF_MATRIX_FLIP_V          0     // 1 = raw row 0 is physically the BOTTOM
#define TOF_MATRIX_FLIP_H          0     // 1 = raw col 0 is the robot's RIGHT
// Speckle rejection (field 2026-08-01): a parked base flapped BLOCKED/clear ~600
// times in 7 min with >1 m genuinely clear ahead — single-zone phantom near returns
// (VL53L7CX speckle) dipped under the at-rest 0.10 m stop floor, and the two-frame
// attack confirm passed them because consecutive speckle frames agree within
// ATTACK_CONFIRM. A real obstacle near enough to matter subtends MULTIPLE zones of
// the 4x8 half-grid (a chair leg at 0.3 m spans several rows even if one column),
// so each half now publishes its SECOND-nearest qualifying zone: one lone screaming
// zone is discarded as speckle. Deliberate coverage trade: an object thin enough to
// light exactly one zone (a bare cable end-on) is invisible — docs §6.3 already
// disclaims full coverage of thin obstacles.
#define TOF_MATRIX_MIN_OBSTACLE_ZONES 2  // qualifying zones per half before it counts

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

// ---- Physical turn verification (LSM6DS3 gyro + encoder fallback) ----------
// Encoder odometry alone cannot distinguish chassis rotation from tyres scrubbing
// in place. When the IMU is healthy, finite turns close on integrated gyro yaw
// instead. Encoder angle remains the fallback when the IMU is absent. A verified
// turn that cannot make physical yaw progress is aborted after a generous,
// command-size-aware timeout instead of grinding indefinitely.
#define TURN_IMU_VERIFY_ENABLED       1
#define TURN_VERIFY_TIMEOUT_MIN_MS 8000
#define TURN_VERIFY_TIMEOUT_MULT    3.0f
#define TURN_VERIFY_TOLERANCE_DEG   1.5f

// (The stalled-pivot "wiggle" assist that lived here — auto-converting a stalled
// pivot into alternating rolling arcs — was removed 2026-07-13 at the owner's
// request: turns should just be turns. If a pivot stalls under full weight,
// that's the torque ceiling talking; the fix is mechanical, not choreography.)

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

// ---- Battery gauge (coulomb counter + LiFePO4 voltage anchors) --------------
// SOC = coulomb-counted discharge against pack capacity, persisted to ESP32 NVS
// so it survives power-off (the ESP32 is USB-powered from the Mac — it goes DARK
// while the pack charges, and the charger clips to the pack terminals so charge
// current never crosses the shunt). Every boot therefore RECONCILES against rest
// voltage: >= FULL_ANCHOR at rest -> 100% (the normal off->charge->on cycle
// self-corrects); on the flat plateau -> trust the saved ledger; below a knee ->
// clamp DOWN (never over-promise). 4S LiFePO4 rest-voltage knees; "rest" = no
// motor drive (idle electronics ~1 A = C/40 on this pack, sag is negligible).
#define BATT_CAPACITY_MAH        40000   // 2x 12.8 V 20 Ah in parallel
#define BATT_SOC_FULL_ANCHOR_MV  13350   // at/above this at rest = charged full
#define BATT_SOC_KNEE1_MV        12900   // below: clamp SOC to <= KNEE1_PCT
#define BATT_SOC_KNEE1_PCT       20
#define BATT_SOC_KNEE2_MV        12500   // below: clamp SOC to <= KNEE2_PCT
#define BATT_SOC_KNEE2_PCT       8
#define BATT_SOC_QUIET_MA        1500    // |current| below this counts as "rest".
                                         // 2500 counted Rex TALKING/GESTURING (~2 A)
                                         // as rest; with the pack's ~160 mΩ junction
                                         // that load sags 12.9 V-looking readings out
                                         // of a healthy plateau -> phantom knee clamp
                                         // (field 2026-07-17: 85% -> 20% mid-speech).
                                         // 1500 still covers idle electronics (~1.3 A).
#define BATT_SOC_ANCHOR_TICKS    20      // consecutive quiet 1 Hz ticks before anchoring
#define BATT_PACK_IR_MOHM        160     // effective pack+junction resistance: rest
                                         // voltage is estimated as mv + ma*IR so the
                                         // sag under idle draw doesn't skew anchors
                                         // (measured ~160 mΩ, 2026-07 charging bench)
#define BATT_SOC_SAVE_DELTA_MAH  200     // persist to NVS every 0.5% of capacity...
#define BATT_SOC_SAVE_SECS       600     // ...or at least every 10 min (NVS wear-safe)

// ---- Charging lockout (battery.cpp detect, control.cpp gate) ----------------
// Plugged into the bench supply/charger = the pack current goes NEGATIVE
// (+ = discharging). While charging, ALL drive is locked out — including manual
// and the R3 sensor-bypass — so the base can never roll away on the cord.
// Debounced on the 1 Hz battery tick; both edges emit a "charging" event.
#define BATT_CHARGE_DETECT_MA    250   // sustained charge current at/above this = on charger
#define BATT_CHARGE_DETECT_MV  14000   // ENTER: charger holds ~14.2V (clearly above rest)
// EXIT is decided by CURRENT alone (battery.cpp): sustained discharge at/above
// BATT_CHARGE_EXIT_DISCHARGE_MA proves the pack — not a charger — is carrying the
// load. The old voltage hold floor (13.60V, "stay locked at/above") is GONE:
// a freshly-topped pack rests at ~13.61V surface charge for tens of minutes, so
// the voltage clause kept the drive locked long after a genuine unplug (field
// 2026-07-31, batt at 13.61V, 10 mV above the floor). Voltage windows can't
// separate charger-under-load (~13.7-13.8V) from a fresh pack's rest (~13.6V);
// current can: taper/cutoff sits near 0 mA, an unplugged pack always feeds the
// electronics. TUNE: with the robot OFF and the charger unplugged, read the idle
// discharge in the battery menubar — EXIT_DISCHARGE_MA must sit clearly BELOW
// that idle draw and clearly ABOVE the charger's taper noise (~0±50 mA).
#define BATT_CHARGE_ENTER_TICKS    3   // ~3 s of charge current before locking out
#define BATT_CHARGE_EXIT_DISCHARGE_MA 100 // sustained discharge >= this = unplugged
                                          // (was 250 — above the ESP32+sensors idle
                                          // draw, so a resting unplug never released)
#define BATT_CHARGE_EXIT_TICKS    90   // ~90 s of sustained discharge before releasing.
                                       // Was 8 s — field 2026-08-07 18:19: the host's
                                       // STARTUP burst (servos+audio) out-drew the bench
                                       // supply through the ~160 mΩ junction for 8+ s,
                                       // faking an unplug while the cable was attached;
                                       // 3 min later the flinch reflex rolled the base
                                       // on the cord. A plugged pack can only sustain
                                       // apparent discharge as long as a load spike
                                       // lasts; a REAL unplug discharges for minutes.
                                       // Cost: after a genuine unplug the drive stays
                                       // locked ~90 s. The counter needs CONSECUTIVE
                                       // discharge ticks, so equilibrium (net ~0 mA,
                                       // charger carrying the load) resets it.

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
#define ZONE_SPEED_REF_MS     0.80f   // speed at which the configured zones fully apply
                                      // (tracks the hardwood-mode ceiling; carpet mode
                                      // tops out higher but is drag-limited in practice)

// ---- Finite-command reflex-block grace (control.cpp) ------------------------
// A finite move/come used to terminate done:blocked the instant the reflex latched
// BLOCKED in its travel direction — so a single phantom near frame killed the whole
// command ("move forward 3 feet" refused with >1 m genuinely clear, field
// 2026-08-01). The reflex still zeroes velocity toward the block IMMEDIATELY
// (safety unchanged); the command itself now survives a block that clears within
// this window and resumes on its own. Only a block that persists — a real wall or
// person — gives up with done:blocked. Sized well above the ToF flap period
// (~150-225 ms per phantom episode) and below "feels stuck" for the operator.
#define FINITE_BLOCK_GRACE_MS  900

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
// TELEOP SURFACE MODES — the gamepad's own caps, in REAL m/s / rad/s, independent
// of params.max_lin/max_ang (the AUTONOMOUS limits the Mac pushes down on connect).
// control_tick clamps a MANUAL drive to the larger (carpet) profile; autonomous
// motion stays on the params caps.
//
// History: launched as 3 abstract speed LEVELS (slow/med/full, L3 cycling), retuned
// repeatedly (bare base -> full weight -> carpet -> real units). At full build
// weight the whole 3-level ladder was insufficient (owner 2026-07-11: full stick on
// level 3 struggles on carpet, turning difficult even on hardwood) — what actually
// varies is the SURFACE, so L3 now TOGGLES two surface profiles instead:
//   HARDWOOD (boot/reconnect default): a bit above the old level-3 top speed.
//   CARPET: maximum authority — higher speed ceiling and a full-saturation spin
//     breakaway kick. NOTE the measured torque ceiling stands: an in-place spin on
//     the test carpet saturated both wheels at 1023 duty with ~zero rotation, so
//     carpet mode makes turning as strong as the hardware allows (arcs improve;
//     pure pivots on deep pile may still stall — that's motors, not firmware).
#define GAMEPAD_HARDWOOD_LIN_MS    0.25f   // ⚠ TEMP CAP 2026-07-17 (was 0.80): the 60RPM
                                           // left motor tops out ~0.31 m/s no-load —
                                           // commanding beyond it makes the faster right
                                           // motor veer the base. Restore when the
                                           // matching right motor is fitted.
// Pure-pivot kinematics are wheel_speed = angular_rate * track_width / 2. BOTH
// modes command the full wheel-speed ceiling for turns (owner 2026-07-13): the
// loaded chassis needs every bit of scrub authority regardless of surface, and
// the stick still scales below the ceiling for fine control.
#define GAMEPAD_HARDWOOD_ANG_RADS  ((2.0f * WHEEL_TARGET_MAX_MS) / TRACK_WIDTH_M)
#define GAMEPAD_HARDWOOD_SPIN_KICK 750.0f  // stall-gated pivot breakaway (see below)
#define GAMEPAD_HARDWOOD_SPIN_RUN  380.0f  // pivot RUNNING floor (sustained scrub carry)
#define GAMEPAD_CARPET_LIN_MS      0.28f   // ⚠ TEMP CAP 2026-07-17 (was 1.05) — see above
#define GAMEPAD_CARPET_ANG_RADS    ((2.0f * WHEEL_TARGET_MAX_MS) / TRACK_WIDTH_M)
#define GAMEPAD_CARPET_SPIN_KICK   1023.0f // full saturation — everything the bridge has
#define GAMEPAD_CARPET_SPIN_RUN    650.0f  // pile drag needs most of the range sustained
// Forward/back stick RESPONSE CURVE: lin command = sign(fwd)*|fwd|^GAMMA * level max.
// GAMMA < 1 is concave ("anti-expo"): more authority at small stick pushes — at 25%
// stick you command ~|0.25|^0.6 ≈ 44% of the level's max (linear gave 25%) — while full
// deflection still hits exactly the level max, so top speeds are unchanged. This is what
// makes the loaded base actually MOVE at small/medium stick without a linear speed-up.
// 1.0 = linear (old feel). Applies to the LINEAR axis only; the turn axis stays linear
// and the spin↔arcade blend keys off the RAW stick so the tuned blend bands don't shift.
#define GAMEPAD_LIN_GAMMA      0.60f
// (GAMEPAD_SPIN_SCALE retired 2026-07-11 with the speed levels: turn authority is now
// always the surface mode's full ang ceiling; the blend below only morphs the wheel
// MIXING between spin-in-place and arcade arc.)
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
// (GAMEPAD_FULL_OVERRIDE_FRAC retired 2026-07-16: hold-L2+R2 override replaced by the
//  R3-toggled sensor-bypass mode — see gamepad.cpp header.)
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
