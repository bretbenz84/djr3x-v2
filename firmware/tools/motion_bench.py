#!/usr/bin/env python3
"""DJ-R3X motion bench tool — drive-base bring-up + calibration over USB serial.

Talks the Mac<->ESP32 wire protocol (docs/motion_protocol.md v1) to a board running
the LIVE firmware (built with -DMOTION_HW_PRESENT=1). Run the subcommands roughly in
this order as you wire and verify the base. Reuses the smoke-test serial client.

  bringup            Guided run of all four stages below in order, with a safety gate
                     before each — the easiest way to bring up a fresh base.

  encoder            Motors stay OFF. Streams odometry so you HAND-TURN a wheel and
                     confirm the encoder reads + which way it counts (per wheel:
                     ENC_SIGN_L / ENC_SIGN_R in calib.h). Safe any time.

  wheel SIDE         ON A STAND (wheels off the ground). Powers ONE wheel (left/right/
                     both) open-loop at a fixed duty for a couple seconds — no PID, no
                     kinematics — so you can confirm each motor is wired to the right
                     side and spins the right way BEFORE trusting the closed loop.
                     Reads the encoder speed back and prints a wiring decision table.

  spin               ON A STAND (wheels off the ground). Gentle forward drive (~3s)
                     with a runaway guard that auto-ESTOPs if the closed loop diverges
                     (a motor/encoder sign mismatch). Confirms the motor(s) drive
                     forward under PID. Requires the encoder(s) wired + verified first.

  straight [--dist]  ON THE FLOOR, clear space, hand near estop. `move` forward a fixed
                     distance, then report measured x (distance calibration) and theta
                     drift (wheel match / straight-line tracking).

  turn [--deg]       ON THE FLOOR. `turn` in place, then report measured theta vs
                     commanded (track-width calibration).

Calibration (docs/motion_system.md §14):
  - distance:  run `straight --dist 1.0`; set COUNTS_PER_METER *= x_measured / 1.0
  - track:     run `turn --deg 360`;     set TRACK_WIDTH_M  *= theta_measured / 2π
Then rebuild + reflash the live firmware.
"""
import argparse
import math
import os
import sys
import time

_TOOLS_DIR = os.path.dirname(os.path.abspath(__file__))
_REPO_ROOT = os.path.dirname(os.path.dirname(_TOOLS_DIR))
sys.path.insert(0, _TOOLS_DIR)
from motion_serial_smoketest import MotionClient  # noqa: E402


def _default_port() -> "str | None":
    """The board's port from .env (MOTION_ESP32_PORT) — the same source main.py uses,
    so the bench tool needs no --port on a configured machine. Override with --port."""
    try:
        from dotenv import load_dotenv
        load_dotenv(os.path.join(_REPO_ROOT, ".env"))
    except Exception:
        pass
    return os.getenv("MOTION_ESP32_PORT", "").strip() or None


DEFAULT_PORT = _default_port()


def odom_line(t):
    o = (t or {}).get("odom", {})
    return (f"state={str((t or {}).get('state')):>8}  x={o.get('x', 0):+.3f}  "
            f"y={o.get('y', 0):+.3f}  theta={o.get('theta', 0):+.3f}  "
            f"lin={o.get('lin', 0):+.3f}  ang={o.get('ang', 0):+.3f}")


def confirm(msg, skip):
    if skip:
        return True
    try:
        return input(f"{msg} [y/N] ").strip().lower() == "y"
    except EOFError:
        return False


def cmd_encoder(c, _args):
    print("ENCODER TEST — motors stay OFF. Hand-roll a wheel and watch x/theta:")
    print("  roll FORWARD -> x and theta climb; reverse -> they drop.")
    print("  Wrong way when you roll forward -> flip that wheel's ENC_SIGN_* in calib.h.")
    print("  Ctrl-C to stop.\n")
    try:
        while True:
            print("  " + odom_line(c.telemetry()), end="\r")
            time.sleep(0.15)
    except KeyboardInterrupt:
        print()


def _wheel_jog(c, side, frac, secs):
    """Jog ONE wheel open-loop for `secs`, watching the encoder-derived speed for that
    side. Returns (peak_signed_speed, peak_other_side) in m/s. Prints live vl/vr."""
    key = "vl" if side == "left" else "vr"
    other = "vr" if side == "left" else "vl"
    direction = "FORWARD" if frac >= 0 else "REVERSE"
    print(f"\n  Jogging the {side.upper()} wheel {direction} "
          f"(frac={frac:+.2f}) for {secs:.1f}s — WATCH THAT WHEEL.")
    c.clear()
    seq = c.send({"cmd": "wheel", "side": side, "frac": frac, "ms": int(secs * 1000)})
    ack = c.wait_for(lambda m: m.get("type") == "ack" and m.get("seq") == seq, 2.0)
    if not (ack and ack.get("accepted")):
        print(f"    NOT accepted: {ack} — clear estop/fault (or release the gamepad) first.")
        return None, None
    peak, peak_other = 0.0, 0.0
    end = time.time() + secs + 0.4
    while time.time() < end:
        w = (c.telemetry() or {}).get("wheels", {})
        v, vo = w.get(key, 0.0), w.get(other, 0.0)
        if abs(v) > abs(peak):
            peak = v
        if abs(vo) > abs(peak_other):
            peak_other = vo
        print(f"    {key}={v:+.3f} m/s   {other}={vo:+.3f} m/s     ", end="\r")
        time.sleep(0.1)
    c.wait_for(lambda m: m.get("type") == "done" and m.get("seq") == seq, 1.5)
    c.send({"cmd": "stop"})
    print()
    return peak, peak_other


def _wheel_verdict(side, frac, peak, peak_other):
    """Print the wiring/direction decision from the measured encoder speeds."""
    moved = abs(peak) > 0.015
    other_moved = abs(peak_other) > 0.015
    fwd_cmd = frac >= 0
    print(f"    measured: this wheel peak={peak:+.3f} m/s, other wheel={peak_other:+.3f} m/s")
    if other_moved and not moved:
        print(f"    ✗ WRONG SIDE: the OTHER wheel moved, not the {side}. The two motors'")
        print("      PWM leads are swapped — fix the wiring (PIN_L_* vs PIN_R_* in pins.h).")
        return
    if not moved:
        print("    ✗ NO MOTION on this wheel (encoder read ~0). Either:")
        print("        • did it spin BY EYE? yes -> the ENCODER isn't reading: check its")
        print("          3.3V/GND + A/B pins (PIN_ENC_* in pins.h).")
        print("        • no -> the MOTOR isn't driving: check 12V on B+/B-, the enable pin")
        print("          (R_EN+L_EN -> PIN_*_EN), the RPWM/LPWM leads, and a common ground.")
        return
    # It moved and the encoder read it. Direction: a FORWARD command should read +.
    forward_reading = (peak > 0) == fwd_cmd
    print(f"    ✓ this wheel spun and its encoder read it ({'+' if peak > 0 else '−'} counts).")
    print("      Confirm BY EYE which way it physically turned:")
    if forward_reading:
        print(f"        • spun {'FORWARD' if fwd_cmd else 'REVERSE'} (as commanded) -> motor +")
        print("          encoder agree for this wheel. ✓ Done, nothing to change.")
        print(f"        • spun the OTHER way -> motor AND encoder are BOTH flipped: swap this")
        print("          wheel's M+/M- leads (or flip MOTOR_SIGN_*) AND flip its ENC_SIGN_*.")
    else:
        print("        • spun the way you commanded BY EYE -> the ENCODER sign is backward:")
        print(f"          flip ENC_SIGN_{'L' if side == 'left' else 'R'} in calib.h.")
        print("        • spun the OPPOSITE way BY EYE -> the MOTOR polarity is backward:")
        print(f"          swap this wheel's M+/M- leads (or flip MOTOR_SIGN_"
              f"{'L' if side == 'left' else 'R'}).")
    print("      (Get each wheel: spins FORWARD by eye AND reads + here, before `spin`.)")


def cmd_wheel(c, args):
    print("WHEEL WIRING TEST — base ON A STAND (wheels off the ground), hand near estop.")
    print("  Powers one wheel OPEN-LOOP (no PID) at a fixed duty, then reads its encoder.")
    print("  Use it to confirm each motor: right SIDE, right DIRECTION, encoder agrees.\n")
    frac = -abs(args.frac) if args.reverse else abs(args.frac)
    sides = ["left", "right"] if args.side == "both" else [args.side]
    if not confirm(f"Wheels OFF THE GROUND and ready to power {'/'.join(sides)}?", args.yes):
        return
    for i, side in enumerate(sides):
        if i > 0 and not confirm(f"Ready to test the {side} wheel?", args.yes):
            break
        peak, peak_other = _wheel_jog(c, side, frac, args.secs)
        if peak is not None:
            _wheel_verdict(side, frac, peak, peak_other)
    c.send({"cmd": "stop"})


def cmd_spin(c, _args):
    print("SPIN TEST — base ON A STAND (wheels off the ground), hand near power/estop.")
    print("  Gentle forward drive for ~3s with a runaway guard.\n")
    c.clear()
    end, saw, aborted = time.time() + 3.0, False, False
    while time.time() < end:
        c.send({"cmd": "drive", "lin": 0.08, "ang": 0.0})   # re-send within the 300ms deadman
        time.sleep(0.1)
        t = c.telemetry() or {}
        o = t.get("odom", {})
        lin, ang = o.get("lin", 0.0), o.get("ang", 0.0)
        print("  " + odom_line(t), end="\r")
        if abs(lin) > 0.02:
            saw = True
        if abs(lin) > 0.40 or lin < -0.05 or abs(ang) > 2.0:
            c.send({"cmd": "estop"})
            aborted = True
            print("\n  !! RUNAWAY / WRONG DIRECTION -> auto-ESTOP.")
            print("     The encoder sign was validated already, so this is motor polarity:")
            print("     SWAP that motor's M+/M- leads on the BTS7960 (and re-run).")
            break
    c.send({"cmd": "stop"})
    print()
    if aborted:
        time.sleep(0.3)
        c.send({"cmd": "clear"})
    elif saw:
        print("  OK: spun and stayed stable. Confirm BY EYE the wheel(s) spun FORWARD.")
    else:
        print("  No motion. Check 12V on B+/B-, R_EN+L_EN->enable pin, RPWM/LPWM pins,")
        print("  common ground, and the motor leads on M+/M-.")


def _run_finite(c, cmd, seq_payload, settle_timeout, label):
    c.clear()
    seq = c.send(seq_payload)
    ack = c.wait_for(lambda m: m.get("type") == "ack" and m.get("seq") == seq, 2.0)
    if not (ack and ack.get("accepted")):
        print(f"  {label} NOT accepted: {ack}")
        if (ack or {}).get("reason") == "manual_override":
            print("    (gamepad owns the base — press Start on the pad, or power-cycle the ESP32)")
        return None
    done = c.wait_for(lambda m: m.get("type") == "done" and m.get("seq") == seq, settle_timeout)
    if not done:
        c.send({"cmd": "stop"})
        print(f"  {label} did not complete in {settle_timeout:.0f}s (encoders wired/turning?).")
        return None
    result = done.get("result", "completed")
    if result != "completed":
        print(f"  NOTE: {label} ended early — done result: {result!r}"
              + (" (obstacle reflex stopped it)" if result == "blocked" else ""))
    return done


def _current_params(c):
    """The board's live effective params (an empty `config` echoes them)."""
    seq = c.send({"cmd": "config"})
    ack = c.wait_for(lambda m: m.get("type") == "ack" and m.get("seq") == seq, 2.0)
    return (ack or {}).get("config", {})


def _ask_float(prompt):
    """Prompt for a number; empty/EOF/garbage -> None (skip the calibration math)."""
    try:
        raw = input(prompt).strip()
    except EOFError:
        return None
    if not raw:
        return None
    try:
        return float(raw)
    except ValueError:
        print(f"  (couldn't parse {raw!r} — skipping)")
        return None


def cmd_straight(c, args):
    print("Mark the floor at a fixed reference point on the base BEFORE driving —")
    print("the calibration math needs the TAPE-MEASURED physical distance, not odometry.")
    if not confirm(f"Robot will DRIVE FORWARD ~{args.dist} m on the floor. Ready?", args.yes):
        return
    print(f"\nSTRAIGHT — move dist={args.dist} m:")
    done = _run_finite(c, "move", {"cmd": "move", "dist": args.dist, "speed": 0.12},
                       settle_timeout=max(8.0, args.dist / 0.12 + 6.0), label="move")
    c.send({"cmd": "stop"})
    if not done:
        return
    o = done.get("odom", {})
    x, th = o.get("x"), o.get("theta")
    print(f"  done. odometry x={x:+.3f} m (cmd {args.dist} m), theta drift={th:+.3f} rad")
    if th is not None and abs(th) > 0.05:
        print("    veered — left/right wheels mismatched (PID gains or an ENC_SIGN_* flip).")
    # Calibration: odometry-vs-TAPE from the SAME run (works even if the run was cut
    # short by the reflex — both numbers cover the same wheel travel). NOT x/dist:
    # odometry x lands on the commanded dist by construction, so that ratio is ~1.
    phys = _ask_float("  Tape-measured PHYSICAL distance traveled (m, empty to skip): ")
    if phys and phys > 0 and x:
        cur = _current_params(c).get("counts_per_meter")
        if cur:
            new_cpm = float(cur) * float(x) / phys
            print(f"    counts_per_meter: {float(cur):.1f} -> {new_cpm:.1f}"
                  f"  (odometry {x:.3f} m vs tape {phys:.3f} m)")
            print(f"    push live now:   motion_bench.py set --counts-per-meter {new_cpm:.1f}")
            print("    persist:         calib.h COUNTS_PER_METER (reflash) or .env MOTION_COUNTS_PER_METER")
        else:
            print("    (no config echo from the board — compute by hand: cpm_new = cpm_old * odom/tape)")


def cmd_turn(c, args):
    print("Mark which way the base faces BEFORE spinning (floor arrow + a mark on the")
    print("base) — the calibration math needs the PHYSICALLY-observed rotation. Do the")
    print("`straight` distance calibration FIRST: heading odometry scales with cpm too.")
    if not confirm(f"Robot will SPIN {args.deg}° in place on the floor. Ready?", args.yes):
        return
    print(f"\nTURN — turn deg={args.deg} (+ = left/CCW):")
    done = _run_finite(c, "turn", {"cmd": "turn", "deg": args.deg, "rate": 45},
                       settle_timeout=max(8.0, abs(args.deg) / 45.0 + 6.0), label="turn")
    c.send({"cmd": "stop"})
    if not done:
        return
    th = done.get("odom", {}).get("theta")
    print(f"  done. odometry theta={th:+.3f} rad (cmd {args.deg}°)")
    print("  (odometry theta WRAPS to (-pi, pi] — after a full turn it reads ~0; that's")
    print("   why the tape/eye measurement below is the truth, not this number.)")
    # Calibration: the turn completed when ODOMETRY progress hit the commanded angle,
    # so the wheels rolled exactly enough counts for cmd° at the CURRENT track width.
    # If the base physically rotated R°, then track_true = track_cur * cmd / R.
    phys = _ask_float(f"  PHYSICALLY-observed rotation (degrees, e.g. 270; empty to skip): ")
    if phys and phys != 0:
        cur = _current_params(c).get("track_width_m")
        if cur:
            new_track = float(cur) * abs(args.deg) / abs(phys)
            print(f"    track_width_m: {float(cur):.4f} -> {new_track:.4f}"
                  f"  (commanded {abs(args.deg):.0f}° vs physical {abs(phys):.0f}°)")
            print(f"    push live now:   motion_bench.py set --track-width {new_track:.4f}")
            print("    persist:         calib.h TRACK_WIDTH_M (reflash) or .env MOTION_TRACK_WIDTH_M")
        else:
            print("    (no config echo — compute by hand: track_new = track_old * cmd_deg/phys_deg)")


_PARAM_ORDER = ("kp", "ki", "kd", "kff", "min_duty", "accel_lin", "accel_ang",
                "counts_per_meter", "track_width_m",
                "max_lin", "max_ang", "slow_zone_m", "stop_zone_m",
                "assist_enabled", "assist_engage_mm", "assist_gain")


def _print_params(cfg):
    if not cfg:
        print("  (no config echoed)")
        return
    for k in _PARAM_ORDER:
        if k in cfg:
            print(f"    {k:18} = {cfg[k]}")


def cmd_show(c, _args):
    """Read back the ESP32's current effective params (an empty config echoes them)."""
    seq = c.send({"cmd": "config"})
    ack = c.wait_for(lambda m: m.get("type") == "ack" and m.get("seq") == seq, 2.0)
    if not ack:
        print("  no config ack"); return
    print("Current effective ESP32 params:")
    _print_params(ack.get("config", {}))


def cmd_set(c, args):
    """Push PID gains / calibration live (runtime only). Persist by recording the
    winning values in config.py / .env (Mac pushes on connect) or calib.h."""
    keys = {}
    if args.kp is not None: keys["kp"] = args.kp
    if args.ki is not None: keys["ki"] = args.ki
    if args.kd is not None: keys["kd"] = args.kd
    if args.kff is not None: keys["kff"] = args.kff
    if args.min_duty is not None: keys["min_duty"] = args.min_duty
    if args.accel_lin is not None: keys["accel_lin"] = args.accel_lin
    if args.accel_ang is not None: keys["accel_ang"] = args.accel_ang
    if args.max_lin is not None: keys["max_lin"] = args.max_lin
    if args.max_ang is not None: keys["max_ang"] = args.max_ang
    if args.counts_per_meter is not None: keys["counts_per_meter"] = args.counts_per_meter
    if args.track_width is not None: keys["track_width_m"] = args.track_width
    if args.assist is not None: keys["assist_enabled"] = bool(args.assist)
    if args.assist_engage_mm is not None: keys["assist_engage_mm"] = args.assist_engage_mm
    if args.assist_gain is not None: keys["assist_gain"] = args.assist_gain
    if not keys:
        print("  nothing to set — pass --kp/--ki/--kd/--kff/--min-duty/--accel-lin/"
              "--accel-ang/--max-lin/--max-ang/--counts-per-meter/--track-width")
        return
    # Calibration (geometry) re-scales odometry immediately, which would corrupt an
    # in-flight finite command's progress — only allow it at idle. Gains are safe live.
    geom = sorted(k for k in keys if k in ("counts_per_meter", "track_width_m"))
    if geom:
        st = (c.telemetry() or {}).get("state")
        if st != "idle":
            print(f"  refusing to change calibration ({', '.join(geom)}) while state={st!r}.")
            print("  Stop the base first — a geometry change mid-motion re-scales the command.")
            return
    print(f"Pushing {keys} …")
    seq = c.send({"cmd": "config", **keys})
    ack = c.wait_for(lambda m: m.get("type") == "ack" and m.get("seq") == seq, 2.0)
    if not ack:
        print("  no config ack"); return
    if ack.get("reason") == "clamped":
        print("  (some values were CLAMPED to safe ranges — effective values below)")
    print("Effective ESP32 params after set:")
    _print_params(ack.get("config", {}))
    print("  NOTE: runtime-only — lost on ESP32 reboot. To persist, record these in")
    print("        config.py / .env (MOTION_WHEEL_KP/KI/KD, MOTION_COUNTS_PER_METER,")
    print("        MOTION_TRACK_WIDTH_M) or bake them into firmware/djr3x_motion/calib.h.")


def cmd_bringup(c, args):
    """Guided one-shot bring-up + calibration: encoder -> spin -> straight -> turn,
    with a safety gate before each stage so the base can be repositioned. Ctrl-C aborts
    (the base is stopped in main's finally). Each stage reuses the standalone subcommand."""
    print("=" * 72)
    print("GUIDED DRIVE-BASE BRING-UP — stages in order (docs/motion_system.md §14,§15).")
    print("Reposition the base when prompted; Ctrl-C aborts and stops the base.")
    print("=" * 72)

    print("\n[1/5] ENCODER DIRECTION — motors stay OFF (safe any time).")
    if confirm("Start the encoder hand-roll check?", args.yes):
        cmd_encoder(c, args)            # loops until Ctrl-C, then returns
    print("  -> If a wheel counted the WRONG way rolling forward, flip its ENC_SIGN_* in")
    print("     calib.h (or push live) BEFORE the powered stages.")

    print("\n[2/5] WHEEL WIRING — one wheel at a time, OPEN-LOOP (no PID), OFF THE GROUND.")
    print("  Confirms each motor is on the right side and spins the right way.")
    if confirm("Wheels off the ground and ready to power each wheel briefly?", False):
        for side in ("left", "right"):
            peak, peak_other = _wheel_jog(c, side, 0.35, 1.5)
            if peak is not None:
                _wheel_verdict(side, 0.35, peak, peak_other)
    else:
        print("  skipped.")

    print("\n[3/5] SPIN UNDER PID — wheels must be OFF THE GROUND (on a stand).")
    if confirm("Wheels off the ground and ready to drive the motors?", False):
        cmd_spin(c, args)
    else:
        print("  skipped.")

    print(f"\n[4/5] STRAIGHT — place the base ON THE FLOOR, ~{args.dist} m clear ahead.")
    cmd_straight(c, args)               # has its own ready? prompt

    print(f"\n[5/5] TURN — clear space to spin {args.deg}° in place on the floor.")
    cmd_turn(c, args)                   # has its own ready? prompt

    print("\n" + "=" * 72)
    print("BRING-UP DONE. To make calibration survive an ESP32 reboot, take the scale")
    print("factors printed above and either:")
    print("  - push live + record in .env / config.py: MOTION_COUNTS_PER_METER,")
    print("    MOTION_TRACK_WIDTH_M, MOTION_WHEEL_KP/KI/KD (Rex re-pushes on connect), or")
    print("  - bake them into firmware/djr3x_motion/calib.h and reflash.")
    print(f"  Read the live effective values any time:  motion_bench.py show")
    print("=" * 72)


def cmd_wheels(c, _args):
    """Stream the per-wheel drive diagnostic (firmware telemetry `wheels`). Drive
    STRAIGHT on the gamepad and read vl/vr (measured wheel speed, m/s) + dl/dr
    (commanded duty):
      dl~dr but vl<vr  -> left drivetrain physically weaker; the PID isn't
                          compensating yet (soft tune) — raise kp/kd, measure cpm.
      dl>dr and vl~vr  -> PID IS compensating (more duty to the lagging wheel);
                          the ramp lag is the transient before the loop catches up.
      vl~vr but it physically veers -> an encoder is mis-scaled (shared-cpm issue)."""
    print("PER-WHEEL DIAGNOSTIC — drive straight on the gamepad; watch vl/vr and dl/dr.")
    print("  Ctrl-C to stop.\n")
    try:
        while True:
            t = c.telemetry() or {}
            o, w = t.get("odom", {}), t.get("wheels", {})
            vl, vr = w.get("vl", 0.0), w.get("vr", 0.0)
            print(f"  {str(t.get('state')):>8}  lin={o.get('lin', 0):+.3f} theta={o.get('theta', 0):+.3f}"
                  f"  vl={vl:+.3f} vr={vr:+.3f} (d={vr - vl:+.3f})"
                  f"  dl={int(w.get('dl', 0)):+5d} dr={int(w.get('dr', 0)):+5d}", end="\r")
            time.sleep(0.13)
    except KeyboardInterrupt:
        print()


def main():
    ap = argparse.ArgumentParser(description="DJ-R3X motion bench / calibration tool")
    ap.add_argument("--port", default=DEFAULT_PORT,
                    help="serial device (default: MOTION_ESP32_PORT from .env)")
    ap.add_argument("--yes", action="store_true", help="skip the floor-test confirmation prompt")
    sub = ap.add_subparsers(dest="cmd", required=True)
    sub.add_parser("encoder")
    wp = sub.add_parser("wheel")
    wp.add_argument("side", choices=("left", "right", "both"),
                    help="which wheel to power open-loop")
    wp.add_argument("--frac", type=float, default=0.35,
                    help="drive fraction 0..1 of full duty (default 0.35)")
    wp.add_argument("--secs", type=float, default=1.5, help="run time per wheel (default 1.5)")
    wp.add_argument("--reverse", action="store_true", help="jog REVERSE instead of forward")
    sub.add_parser("spin")
    sub.add_parser("wheels")
    sp = sub.add_parser("straight"); sp.add_argument("--dist", type=float, default=1.0)
    tp = sub.add_parser("turn"); tp.add_argument("--deg", type=float, default=360.0)
    bp = sub.add_parser("bringup")
    bp.add_argument("--dist", type=float, default=1.0)
    bp.add_argument("--deg", type=float, default=360.0)
    sub.add_parser("show")
    st = sub.add_parser("set")
    st.add_argument("--kp", type=float); st.add_argument("--ki", type=float)
    st.add_argument("--kd", type=float)
    st.add_argument("--kff", type=float, help="velocity feedforward (duty per m/s)")
    st.add_argument("--min-duty", type=float, dest="min_duty", help="stiction breakaway kick (duty)")
    st.add_argument("--accel-lin", type=float, dest="accel_lin", help="teleop linear slew (m/s^2)")
    st.add_argument("--accel-ang", type=float, dest="accel_ang", help="teleop angular slew (rad/s^2)")
    st.add_argument("--max-lin", type=float, dest="max_lin", help="linear speed cap (m/s)")
    st.add_argument("--max-ang", type=float, dest="max_ang", help="angular speed cap (rad/s)")
    st.add_argument("--counts-per-meter", type=float, dest="counts_per_meter")
    st.add_argument("--track-width", type=float, dest="track_width")
    st.add_argument("--assist", type=int, choices=(0, 1), dest="assist",
                    help="hallway steering assist on/off (manual forward drive)")
    st.add_argument("--assist-engage-mm", type=float, dest="assist_engage_mm",
                    help="walls beyond this (mm) don't steer")
    st.add_argument("--assist-gain", type=float, dest="assist_gain",
                    help="rad/s per meter of left-right wall imbalance")
    args = ap.parse_args()

    handlers = {"encoder": cmd_encoder, "wheel": cmd_wheel, "spin": cmd_spin,
                "wheels": cmd_wheels, "straight": cmd_straight, "turn": cmd_turn,
                "bringup": cmd_bringup, "show": cmd_show, "set": cmd_set}
    if not args.port:
        ap.error("no serial port — set MOTION_ESP32_PORT in .env or pass --port "
                 "(find it with `arduino-cli board list`)")
    print(f"Opening {args.port} …")
    c = MotionClient(args.port)
    time.sleep(1.7)  # board auto-resets on port open; let it boot
    c.clear()
    try:
        handlers[args.cmd](c, args)
    finally:
        c.send({"cmd": "stop"})
        time.sleep(0.2)
        c.close()


if __name__ == "__main__":
    main()
