#!/usr/bin/env python3
"""Host-side protocol smoke test for the DJ-R3X ESP32 motion firmware.

Exercises the Mac<->ESP32 wire contract (docs/motion_protocol.md v1) against a
real board over USB serial and prints a human-readable PASS/FAIL per check. This
is the bring-up acceptance test: it proves the firmware speaks the protocol
correctly with NOTHING wired to the ESP32 (the firmware runs a stubbed plant).

Usage:
    venv/bin/python firmware/tools/motion_serial_smoketest.py [--port /dev/cu.usbserial-10]

Exit code 0 = all checks passed.
"""
import argparse
import json
import sys
import threading
import time

import serial  # pyserial


class MotionClient:
    """Reader thread + background heartbeat over one serial port."""

    def __init__(self, port, baud=115200):
        self.ser = serial.Serial(port, baud, timeout=0.1)
        self.lock = threading.Lock()
        self.messages = []          # all parsed inbound dicts (chronological)
        self.latest_telemetry = None
        self.parse_errors = 0
        self._stop = False
        self._ping_on = True
        self._seq = 0
        self._rx = threading.Thread(target=self._reader, daemon=True)
        self._hb = threading.Thread(target=self._heartbeat, daemon=True)
        self._rx.start()
        self._hb.start()

    # ---- threads ----
    def _reader(self):
        buf = b""
        while not self._stop:
            try:
                chunk = self.ser.read(256)
            except Exception:
                break
            if not chunk:
                continue
            buf += chunk
            while b"\n" in buf:
                line, buf = buf.split(b"\n", 1)
                line = line.strip()
                if not line:
                    continue
                try:
                    obj = json.loads(line.decode("utf-8", "replace"))
                except Exception:
                    self.parse_errors += 1
                    continue
                with self.lock:
                    self.messages.append(obj)
                    if obj.get("type") == "telemetry":
                        self.latest_telemetry = obj

    def _heartbeat(self):
        while not self._stop:
            if self._ping_on:
                self.send({"cmd": "ping"})
            time.sleep(0.15)

    # ---- helpers ----
    def next_seq(self):
        self._seq += 1
        return self._seq

    def send(self, obj):
        if "v" not in obj:
            obj = {"v": 1, **obj}
        if obj.get("cmd") and "seq" not in obj:
            obj["seq"] = self.next_seq()
        line = (json.dumps(obj, separators=(",", ":")) + "\n").encode()
        try:
            self.ser.write(line)
        except Exception:
            pass
        return obj.get("seq")

    def pause_ping(self):
        self._ping_on = False

    def resume_ping(self):
        self._ping_on = True

    def clear(self):
        with self.lock:
            self.messages.clear()

    def wait_for(self, pred, timeout=3.0):
        """Return the first inbound message matching pred(msg), else None."""
        end = time.time() + timeout
        seen = 0
        while time.time() < end:
            with self.lock:
                msgs = self.messages[seen:]
                seen = len(self.messages)
            for m in msgs:
                if pred(m):
                    return m
            time.sleep(0.02)
        return None

    def telemetry(self):
        with self.lock:
            return self.latest_telemetry

    def close(self):
        self._stop = True
        time.sleep(0.2)
        try:
            self.ser.close()
        except Exception:
            pass


# ---- test harness ----
RESULTS = []


def check(name, ok, detail=""):
    RESULTS.append(ok)
    mark = "PASS" if ok else "FAIL"
    print(f"  [{mark}] {name}" + (f"  — {detail}" if detail else ""))


def approx(a, b, tol):
    return a is not None and abs(a - b) <= tol


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--port", default="/dev/cu.usbserial-10")
    ap.add_argument("--baud", type=int, default=115200)
    args = ap.parse_args()

    print(f"Opening {args.port} @ {args.baud} …")
    c = MotionClient(args.port, args.baud)
    # ESP32 typically auto-resets when the port opens; give it time to boot.
    time.sleep(1.5)
    c.clear()

    try:
        # 1) Handshake
        print("\n1) Handshake (hello -> hello)")
        c.send({"cmd": "hello", "host": "smoketest", "proto": 1})
        hello = c.wait_for(lambda m: m.get("type") == "hello", 3.0)
        check("hello reply received", hello is not None, json.dumps(hello) if hello else "no reply")
        if hello:
            check("proto == 1", hello.get("proto") == 1, f"proto={hello.get('proto')}")
            check("advertises caps", isinstance(hello.get("caps"), list) and "drive" in hello.get("caps", []),
                  f"caps={hello.get('caps')}")
            check("has boot_id + fw", "boot_id" in hello and "fw" in hello, f"fw={hello.get('fw')}")

        # 2) Telemetry schema + idle
        print("\n2) Telemetry stream")
        tel = c.wait_for(lambda m: m.get("type") == "telemetry", 2.0)
        check("telemetry received", tel is not None)
        if tel:
            need = ["t", "state", "owner", "gamepad", "zone", "blocked_dir", "cmd_seq",
                    "odom", "tof_mm", "batt_mv", "errs"]
            missing = [k for k in need if k not in tel]
            check("telemetry has all fields", not missing, f"missing={missing}" if missing else "all present")
            check("'fault' key present (null ok)", "fault" in tel, f"fault={tel.get('fault')}")
            check("state is idle at rest", tel.get("state") == "idle", f"state={tel.get('state')}")
            odom = tel.get("odom", {})
            check("odom has x/y/theta/lin/ang", all(k in odom for k in ("x", "y", "theta", "lin", "ang")))
            tof = tel.get("tof_mm", {})
            check("tof_mm has 5 sensors", all(k in tof for k in ("fl", "fc", "fr", "rear", "down")),
                  f"tof={tof}")

        # 3) turn left 90  -> ack + done completed, theta ~ +1.57 (CCW positive)
        print("\n3) turn deg=+90 (left/CCW)")
        c.clear()
        seq = c.send({"cmd": "turn", "deg": 90, "rate": 60})
        ack = c.wait_for(lambda m: m.get("type") == "ack" and m.get("seq") == seq, 2.0)
        check("turn acked accepted", ack is not None and ack.get("accepted") is True, json.dumps(ack))
        done = c.wait_for(lambda m: m.get("type") == "done" and m.get("seq") == seq, 6.0)
        check("turn done:completed", done is not None and done.get("result") == "completed", json.dumps(done))
        if done:
            th = done.get("odom", {}).get("theta")
            check("theta ~ +1.57 rad (left is positive)", approx(th, 1.5708, 0.25), f"theta={th}")

        # 4) move forward 0.3 m -> ack + done completed
        print("\n4) move dist=0.3 m (forward)")
        c.clear()
        seq = c.send({"cmd": "move", "dist": 0.3, "speed": 0.15})
        ack = c.wait_for(lambda m: m.get("type") == "ack" and m.get("seq") == seq, 2.0)
        check("move acked accepted", ack is not None and ack.get("accepted") is True, json.dumps(ack))
        done = c.wait_for(lambda m: m.get("type") == "done" and m.get("seq") == seq, 6.0)
        check("move done:completed", done is not None and done.get("result") == "completed", json.dumps(done))

        # 5) drive deadman: one drive, keep pinging, expect ramp to stop (~<1s)
        print("\n5) drive deadman (one setpoint, no refresh -> ramps to stop)")
        c.clear()
        c.send({"cmd": "drive", "lin": 0.12, "ang": 0.0})
        moving = c.wait_for(lambda m: m.get("type") == "telemetry" and m.get("odom", {}).get("lin", 0) > 0.03, 1.0)
        check("robot started moving", moving is not None,
              f"lin={moving.get('odom',{}).get('lin') if moving else None}")
        time.sleep(0.8)  # deadman is 300ms + ramp
        tel = c.telemetry()
        check("ramped back to stop (lin~0)", approx(tel.get("odom", {}).get("lin"), 0.0, 0.02) if tel else False,
              f"lin={tel.get('odom',{}).get('lin') if tel else None}")

        # 6) heartbeat watchdog: stop all traffic >0.6s -> comms_lost, then recover
        print("\n6) heartbeat watchdog (silence -> comms_lost -> recover)")
        c.clear()
        c.pause_ping()
        time.sleep(0.9)  # watchdog window is 500ms
        tel = c.telemetry()
        cl_evt = c.wait_for(lambda m: m.get("type") == "event" and m.get("event") == "comms_lost", 0.2)
        check("entered comms_lost", (tel and tel.get("state") == "comms_lost") or cl_evt is not None,
              f"state={tel.get('state') if tel else None}")
        c.clear()
        c.resume_ping()
        restored = c.wait_for(lambda m: (m.get("type") == "event" and m.get("event") == "comms_restored")
                              or (m.get("type") == "telemetry" and m.get("state") == "idle"), 2.0)
        check("recovered to idle on next ping", restored is not None, json.dumps(restored))

        # 7) estop precedence: estop -> drive rejected -> clear -> idle
        print("\n7) estop / clear precedence")
        c.clear()
        seq = c.send({"cmd": "estop"})
        ack = c.wait_for(lambda m: m.get("type") == "ack" and m.get("seq") == seq, 2.0)
        check("estop acked", ack is not None and ack.get("accepted") is True, json.dumps(ack))
        tel = c.wait_for(lambda m: m.get("type") == "telemetry" and m.get("state") == "estop", 2.0)
        check("state == estop", tel is not None)
        seq = c.send({"cmd": "drive", "lin": 0.1, "ang": 0})
        ack = c.wait_for(lambda m: m.get("type") == "ack" and m.get("seq") == seq, 2.0)
        check("drive rejected while estop (reason=estop)",
              ack is not None and ack.get("accepted") is False and ack.get("reason") == "estop", json.dumps(ack))
        seq = c.send({"cmd": "clear"})
        ack = c.wait_for(lambda m: m.get("type") == "ack" and m.get("seq") == seq, 2.0)
        check("clear acked accepted", ack is not None and ack.get("accepted") is True, json.dumps(ack))
        tel = c.wait_for(lambda m: m.get("type") == "telemetry" and m.get("state") == "idle", 2.0)
        check("state back to idle after clear", tel is not None)

        # 8) error handling: unknown cmd + bad version
        print("\n8) error handling (unknown_cmd, bad_version)")
        seq = c.send({"cmd": "frobnicate"})
        ack = c.wait_for(lambda m: m.get("type") == "ack" and m.get("seq") == seq, 2.0)
        check("unknown cmd rejected (reason=unknown_cmd)",
              ack is not None and ack.get("reason") == "unknown_cmd", json.dumps(ack))
        seq = c.next_seq()
        c.send({"v": 99, "cmd": "stop", "seq": seq})
        ack = c.wait_for(lambda m: m.get("type") == "ack" and m.get("seq") == seq, 2.0)
        check("bad version rejected (reason=bad_version)",
              ack is not None and ack.get("reason") == "bad_version", json.dumps(ack))

        # 9) clamping: over-cap drive accepted but clamped
        print("\n9) clamping (over-cap drive)")
        c.clear()
        seq = c.send({"cmd": "drive", "lin": 5.0, "ang": 0.0})
        ack = c.wait_for(lambda m: m.get("type") == "ack" and m.get("seq") == seq, 2.0)
        check("over-cap drive accepted+clamped",
              ack is not None and ack.get("accepted") is True and ack.get("reason") == "clamped", json.dumps(ack))
        c.send({"cmd": "stop"})

    finally:
        c.send({"cmd": "stop"})
        time.sleep(0.2)
        passed = sum(1 for r in RESULTS if r)
        total = len(RESULTS)
        print(f"\n=== {passed}/{total} checks passed; parse_errors={c.parse_errors} ===")
        c.close()
        sys.exit(0 if passed == total else 1)


if __name__ == "__main__":
    main()
