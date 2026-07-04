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
        return None
    done = c.wait_for(lambda m: m.get("type") == "done" and m.get("seq") == seq, settle_timeout)
    if not done:
        c.send({"cmd": "stop"})
        print(f"  {label} did not complete in {settle_timeout:.0f}s (encoders wired/turning?).")
        return None
    return done


def cmd_straight(c, args):
    if not confirm(f"Robot will DRIVE FORWARD ~{args.dist} m on the floor. Ready?", args.yes):
        return
    print(f"\nSTRAIGHT — move dist={args.dist} m:")
    done = _run_finite(c, "move", {"cmd": "move", "dist": args.dist, "speed": 0.12},
                       settle_timeout=max(8.0, args.dist / 0.12 + 6.0), label="move")
    if done:
        o = done.get("odom", {})
        x, th = o.get("x"), o.get("theta")
        print(f"  done. measured x={x:+.3f} m (cmd {args.dist} m), theta drift={th:+.3f} rad")
        if x:
            print(f"    distance calibration: COUNTS_PER_METER *= {x / args.dist:.4f}")
        if th is not None and abs(th) > 0.05:
            print("    veered — left/right wheels mismatched (PID gains or an ENC_SIGN_* flip).")
    c.send({"cmd": "stop"})


def cmd_turn(c, args):
    if not confirm(f"Robot will SPIN {args.deg}° in place on the floor. Ready?", args.yes):
        return
    print(f"\nTURN — turn deg={args.deg} (+ = left/CCW):")
    done = _run_finite(c, "turn", {"cmd": "turn", "deg": args.deg, "rate": 45},
                       settle_timeout=max(8.0, abs(args.deg) / 45.0 + 6.0), label="turn")
    if done:
        th = done.get("odom", {}).get("theta")
        want = math.radians(args.deg)
        print(f"  done. measured theta={th:+.3f} rad (cmd {want:+.3f} rad = {args.deg}°)")
        if th:
            print(f"    track-width calibration: TRACK_WIDTH_M *= {want / th:.4f}")
    c.send({"cmd": "stop"})


_PARAM_ORDER = ("kp", "ki", "kd", "kff", "min_duty", "accel_lin", "accel_ang",
                "counts_per_meter", "track_width_m",
                "max_lin", "max_ang", "slow_zone_m", "stop_zone_m")


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

    print("\n[1/4] ENCODER DIRECTION — motors stay OFF (safe any time).")
    if confirm("Start the encoder hand-roll check?", args.yes):
        cmd_encoder(c, args)            # loops until Ctrl-C, then returns
    print("  -> If a wheel counted the WRONG way rolling forward, flip its ENC_SIGN_* in")
    print("     calib.h (or push live) BEFORE the powered stages.")

    print("\n[2/4] SPIN UNDER PID — wheels must be OFF THE GROUND (on a stand).")
    if confirm("Wheels off the ground and ready to drive the motors?", False):
        cmd_spin(c, args)
    else:
        print("  skipped.")

    print(f"\n[3/4] STRAIGHT — place the base ON THE FLOOR, ~{args.dist} m clear ahead.")
    cmd_straight(c, args)               # has its own ready? prompt

    print(f"\n[4/4] TURN — clear space to spin {args.deg}° in place on the floor.")
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
    args = ap.parse_args()

    handlers = {"encoder": cmd_encoder, "spin": cmd_spin, "wheels": cmd_wheels,
                "straight": cmd_straight, "turn": cmd_turn, "bringup": cmd_bringup,
                "show": cmd_show, "set": cmd_set}
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
