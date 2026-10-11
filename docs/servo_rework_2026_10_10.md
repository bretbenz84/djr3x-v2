# Servo rework, October 10 2026: two-servo head, poker elbow and wrist, inverted visor

The owner reworked the head and the poker arm on October 10 2026 and measured the new
positions in the Pololu Maestro Control Center (microseconds; the code uses
quarter-microseconds, x4). The V3 program was updated first and partly verified on the
robot; this is the V2 replication, adapted to V2's own structure. **Every measured
extreme below is a calibrated channel limit, not just a pose. Never widen one without a
new reading from the owner.**

Source of truth for the numbers: `hardware/head_pose.py` (geometry, pure arithmetic) and
`config.SERVO_CHANNELS`. Behaviour: `hardware/servos.py` (driver), `sequences/animations.py`
(lifecycle, beats, looks), `intelligence/consciousness.py` (face following, gaze, idle wander).

## What changed on the robot

| ch | joint | min | max | notes |
|---|---|---|---|---|
| 0 | neck | 1984 | 8960 | unchanged; centre 5472; low pulse looks left |
| 1 | head_lift | 2600 | 7744 | unchanged; level 3600 |
| 2 | head_left | 4032 (1008 µs) | 8128 (2032 µs) | NEW limits; level 5832 (1458 µs); low = head down |
| 3 | visor | 4544 | 6976 | same travel, **inverted**: neutral open 5732 (1433 µs), surprise 4544 (1136 µs), down/park 6976 |
| 4 | hero_elbow | 6300 | 7424 | unchanged |
| 5 | hero_wrist | 1984 | 9984 | unchanged |
| 6 | poker_arm (swing) | 3968 | 8000 | unchanged |
| 7 | hero_arm (mic array) | 3968 | 8000 | unchanged |
| 8–10 | throttle | unchanged | unchanged | unchanged |
| 11 | head_right | 5248 (1312 µs) | 7616 (1904 µs) | NEW; level 6480 (1620 µs); low = head UP |
| 12 | poker_elbow | 3456 (864 µs, down) | 7680 (1920 µs, up) | NEW |
| 13 | poker_wrist | 1984 (496 µs, down) | 9944 (2486 µs, curled in) | NEW; straight 5760 (1440 µs) |

### The two-servo head

The single head-tilt servo is gone. Two servos in the head face each other, each with a
linkage onto one centre post. Driven against each other they **pitch** the head; driven the
same way they shift the post sideways and **cock (tilt)** the head to one side. Owner's
readings (left / right µs): level 1458/1620; fully down 1013/1904; fully up 2032/1312; down
tilted right 1008/1702; down tilted left 1350/1849; up tilted left 2032/1587; up tilted
right 1659/1312.

The code represents the head as `pitch` (−1 fully down, 0 level, +1 fully up) and `tilt`
(−1 cocked to Rex's own left, 0 straight, +1 right). `head_pose.head_pulses(pitch, tilt)`
gives the pair; `head_pose.head_pose(pair)` is the inverse (a least-error fit). Pitch
moves the servos along the measured level/up/down line (asymmetric, two branches); a full
tilt shifts both the same way by 1100 q-µs in total, each taking half within its limits
and the other taking what one cannot. This reproduces all seven readings exactly
(`tests/test_head_pose.py`). Tilting right lowers both pulses. **Confirmed on the robot:
tilt_left cocks the head to Rex's own left.**

Rules enforced in `hardware/servos.py` before anything reaches the wire:

- A write naming one head channel without the other is refused (`IncompleteHeadPose`).
- A pair no pitch/tilt within range explains to within 600 q-µs (150 µs) is refused
  (`HeadPairOutsideEnvelope`) — the linkage never saw corners such as both servos high.
- Holds (`begin_head_gesture`), photo leases and baselines treat the two as one unit.
- `move_to` steps the two so they arrive together (a straight line in pulse space between
  two valid pairs stays inside the envelope; equal increments would not).
- Degrees are still the old estimate (full down ≈ 20°, full up ≈ 25°), unverified.

### Poker elbow and wrist

The elbow can rise fully only while the wrist is at or below 1564 µs (6256). Curled further
in, the wrist meets the arm and the ceiling falls: wrist 1564 → elbow max 1920; 1894 → 1807;
2136 → 1709; 2496 → 1597 µs, linear between (`head_pose.poker_elbow_max`). A pair above the
ceiling is refused (`PokerElbowWristCollision`). Every move is staged one joint at a time
(`head_pose.poker_route`, `servos.move_poker`): elbow down to the lower of current and
target, wrist to its target, elbow up — the ceiling only falls as the wrist curls in, so
this path never crosses it. A lone elbow or wrist is completed from the current command.

Poses (elbow / wrist µs): **park** 1920 / 1564 (the owner's "wheel base park", where the arm
stands when switched on); **poker_up** 1500 / 942 (45° up, hand level — the awake pose);
**poker_out** 1057 / 1389; **poker_down** 864 / 1440.

### Visor

The servo runs inverted: low pulses raise it, high pulses lower it. The September
owner read two positions on 2026-10-11: the **neutral open point is 1433 µs (5732)** and the
**surprise position is 1136 µs (4544)**, which is the channel's own top-of-travel limit.
Down/park stays 6976. Every other visor pose derives from the neutral in `config.py`
(`_derive_visor_calibration`), keeping the spacing the first port used: speaking band
neutral −240 … +400 (5492–6132), camera-clear edge neutral +400 (6132, used as the limit
for squints), deliberate captures neutral −320 (5412), idle drift neutral ±350 (5382–6082)
with an occasional half-blink down to the edge. Mood visor targets are offsets from the
neutral. **The pulse where the visor starts to hide the camera is still unread, so the
camera-clear edge is a guess.**

## Behaviour (what Rex's body does)

- **Cold start.** Weight-loaded channels that read 0 (off since power-up) get their
  unpowered rest first, before anything else can target them: hero elbow 6300, head pair
  fully down (4052 / 7616), poker elbow/wrist at park (7680 / 6256). A channel that reads a
  live pulse (the program restarted with the servos powered) keeps that reading instead.
  The Maestro ignores speed limits for the first target to an off channel, so a wrong rest
  makes the joint jump; the first V3 run's assumed elbow-down rest dropped the poker arm.
- **Startup.** Head to level (lift 3600, pair 5832 / 6480) and visor to its neutral open point (5732, 1433 µs) glide
  together while the neck scans and the throttle unfolds; the poker arm goes from park to
  the 45° pose in clearance stages at the slow lifecycle pace (speed 12).
- **Shutdown / sleep.** Visor down 6976, neck centre, lift 2600, head pair fully down, hero
  elbow 6300 in one glide; throttle to park; poker arm back to park in stages (wrist first,
  then elbow up), allowed through the pose latch like the throttle's park. Channels 5–7
  stay. After the settle, the board is read back and any channel still short of its target
  is logged as `ch3@6400->6976`.
- **Drive stance** (`animations.drive_stance`, called from `hardware/motion.py` before a
  base command, released on the firmware's `done`): neck centre, head level and straight,
  hero elbow 7424, poker swing 8000, poker elbow/wrist at park.
- **Named head steps** (`animations.directed_look_pose`): look_down = lift 3200, pitch −0.8
  keeping the tilt; look_up = lift 4200, pitch +0.8; level/center = lift 3600, pitch 0,
  tilt 0; tilt_left / tilt_right = tilt ∓0.8 keeping pitch and neck; tilt_straight = tilt 0.
  look_left / look_right are **relative**: from nearer the centre they go to the 33° pose
  (2944 / 8000); already there or beyond, a person's request goes on to the neck's limit,
  and Rex's own repeated step (`requested=False`, exploration and games) moves nothing.
  Voice: "tilt your head left", "cock your head right", "straighten your head", "level your
  head" parse to these steps and are not photographed or reported.
  Poker steps: `animations.poker_pose("poker_out" | "poker_up" | "poker_down" | "poker_park")`.
- **Expressions** (`head_pose.HEAD_EXPRESSIONS`, offered to the conversation model as body
  beats `agreement_nod`, `head_tilt`, `perplexed`, `head_wiggle`, `sympathetic` with guidance
  on which feeling fits which and to keep most lines plain; the model chooses, events are
  never mapped to one): nod pitch −0.3 then +0.1; tilt one cock of 0.6 held 0.5 s;
  perplexed cock 0.6 one side then 0.6 the other, 0.7 s each, pitched 0.08 down; wiggle
  three quick 0.35 cocks, 0.12 s each; sympathetic one slow 0.45 cock pitched 0.15 down,
  held 1.2 s. All return the head exactly to where it was and hold nothing; "subtle" halves
  the amounts; sided ones cock away from any lean already there.
- **Talking.** The head pair sways slowly together around its anchor: pitch ±0.06 with tilt
  ±0.1 on one smooth sine (period 2.4 s), Maestro speed 20; the visor plays inside
  5492–6132, an angry squint resting at the camera-clear edge (6132) and never past it.
- **Listening.** The "mhm" nod dips the lift and pitches the pair down 0.07, keeping the cock.
- **Idle life.** Head drift pitch ±0.03 and tilt ±0.06 per move; glances away cock 0.1–0.25
  to a random side; room scans carry a random tilt ±0.3; following a face adjusts the pitch
  (a face low in the frame pitches the head down) and keeps the tilt, settling into a gentle
  0.15–0.3 cock for 3–8 s every 10–25 s; the poker elbow/wrist wander around the 45° pose
  (elbow 5200–6800, wrist 2900–4700) in 300–900 steps through the staged move; the visor
  idles in 5382–6082 around its 1433 µs neutral with an occasional half-blink.
- **Camera.** Both head channels count toward "the head moved, the camera view changed";
  small-move thresholds: channel 2 about 450, channel 11 about 300 q-µs
  (`head_pose.HEAD_SMALL`).
- **Model-facing joints.** `head_pose.joint_percents` reports head_pitch (0 down, 50 level,
  100 up), head_tilt (0 left, 50 straight, 100 right) and visor (0 down, 100 up), never the
  raw head servos.
- **Servo menu-bar console** (`tools/rex_servo_menubar.py`): all 14 channels with the limits
  above, each new slider labelled with its direction; header notes say the sliders are
  independent so the console cannot enforce the head envelope, and that the poker
  elbow/wrist need manual clearance like the throttle.

## Verification

Synthetic only, through `venv/bin/python tools/run_lean_checks.py` (one module per
process, serial/network/audio blocked, fake pulses): `head_pose servo_head_pair body_beats
directed_look pride_mode listening_motion servo_speech_emotion servo_manual_override
servo_menubar_throttle wave_back tool_router performance_plan action_router_catalog
regex_routing_guards motion_agency avatar_rig emotion_orchestrator face_tracking
idle_head_wander gaze_engine body_mood object_glance pet_still`. Nothing here has been
run on the robot by this work.

**Still needs the robot (owner):**

- The visor's real endpoints and the pulse where it starts to hide the camera (the
  camera-clear edge, neutral +400, is a guess). Neutral open (1433 µs) and surprise
  (1136 µs) are the owner's 2026-10-11 readings.
- The shutdown timeout cause: the first V3 shutdown on the new head timed out on the
  head/visor lane with nothing parked; the Maestro silently clamping a target outside the
  range stored in its own settings looks exactly like a servo that never arrives, and the
  visor park 6976 (1744 µs) is the prime suspect. The next shutdown's `shutdown park not
  reached: ch…` warning names the channel.
- Whether the linkage binds anywhere inside the pitch/tilt envelope, and how far a 0.8 tilt
  visibly cocks the head.
- Poker-arm clearance of the body and the other arms at the awake pose, the idle band and
  the drive stance.
- Whether the expressions read well at a distance, and the pace of the talking sway.
- Degrees per pitch fraction (20 down / 25 up is the old estimate).
