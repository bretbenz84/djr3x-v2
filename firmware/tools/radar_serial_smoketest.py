#!/usr/bin/env python3
"""Radar-ring serial smoke test — exercises a flashed djr3x_radar board.

Usage:
    venv/bin/python firmware/tools/radar_serial_smoketest.py [--port /dev/cu.usbmodemXXXX]
    venv/bin/python firmware/tools/radar_serial_smoketest.py --expect-targets   # stub build / occupied room

Without --port, the board is resolved the same way the runtime does: by USB
serial number from RADAR_ESP32_SERIAL in .env (falling back to
RADAR_ESP32_PORT). Read-only and sensor-only: this board cannot move anything,
so the test never risks motion. Exit code 0 = all checks passed.

Checks: hello handshake (caps must include "radar" — the drive base also
answers hello), telemetry schema + rate, per-target sanity, unknown-command
and bad-version acks, and a clean parse-error count. With --expect-targets it
also requires at least one fused target within --seconds (always true on the
stub build's synthetic scene; on real hardware, walk in front of a sensor).
"""
from __future__ import annotations

import argparse
import json
import sys
import threading
import time
from pathlib import Path

import serial  # pyserial
from serial.tools import list_ports

_ROOT = Path(__file__).resolve().parents[2]


class RadarClient:
    """Reader thread over one serial port (no heartbeat — the radar board has
    no watchdog to feed). Reusable by other bench tools, MotionClient-style."""

    def __init__(self, port: str, baud: int = 115200):
        self.ser = serial.Serial(port, baud, timeout=0.1)
        self.messages: list[dict] = []
        self.parse_errors = 0
        self._lock = threading.Lock()
        self._stop = threading.Event()
        self._reader = threading.Thread(target=self._read_loop, daemon=True)
        self._reader.start()

    def _read_loop(self) -> None:
        buf = b""
        while not self._stop.is_set():
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
                    msg = json.loads(line.decode("utf-8", "replace"))
                except Exception:
                    self.parse_errors += 1
                    continue
                if isinstance(msg, dict):
                    with self._lock:
                        self.messages.append(msg)

    def send(self, obj: dict) -> None:
        msg = {"v": 1, **obj}
        self.ser.write((json.dumps(msg, separators=(",", ":")) + "\n").encode())

    def clear(self) -> None:
        with self._lock:
            self.messages.clear()

    def wait_for(self, pred, timeout: float = 2.0) -> dict | None:
        deadline = time.time() + timeout
        seen = 0
        while time.time() < deadline:
            with self._lock:
                msgs = self.messages[seen:]
                seen = len(self.messages)
            for m in msgs:
                if pred(m):
                    return m
            time.sleep(0.02)
        return None

    def close(self) -> None:
        self._stop.set()
        self._reader.join(timeout=1.0)
        try:
            self.ser.close()
        except Exception:
            pass


# ── Tiny check harness (motion_serial_smoketest style) ──────────────────────
RESULTS: list[tuple[str, bool, str]] = []


def check(name: str, ok: bool, detail: str = "") -> None:
    RESULTS.append((name, bool(ok), detail))
    print(f"  [{'PASS' if ok else 'FAIL'}] {name}" + (f"  — {detail}" if detail else ""))


def _read_env(key: str) -> str | None:
    """Scrape .env without importing config (no API keys needed here)."""
    env = _ROOT / ".env"
    if not env.exists():
        return None
    for line in env.read_text().splitlines():
        line = line.strip()
        if line.startswith(f"{key}=") and not line.startswith("#"):
            val = line.split("=", 1)[1].strip()
            return val or None
    return None


def _resolve_port() -> str | None:
    sn = _read_env("RADAR_ESP32_SERIAL")
    if sn:
        matches = [p.device for p in list_ports.comports()
                   if (p.serial_number or "").strip().lower() == sn.lower()]
        matches.sort(key=lambda d: (0 if "/cu." in d else 1, d))
        if matches:
            return matches[0]
        print(f"(RADAR_ESP32_SERIAL={sn} not found among attached ports)")
    return _read_env("RADAR_ESP32_PORT")


def main() -> int:
    ap = argparse.ArgumentParser(description="djr3x_radar serial smoke test")
    ap.add_argument("--port", default=None, help="CDC device (default: resolve from .env)")
    ap.add_argument("--baud", type=int, default=115200)
    ap.add_argument("--seconds", type=float, default=6.0,
                    help="target-watch window for --expect-targets")
    ap.add_argument("--expect-targets", action="store_true",
                    help="require >=1 fused target (stub build always has one)")
    args = ap.parse_args()

    port = args.port or _resolve_port()
    if not port:
        print("No port: pass --port or set RADAR_ESP32_SERIAL/RADAR_ESP32_PORT in .env")
        return 1
    print(f"Opening {port} at {args.baud}...")
    c = RadarClient(port, args.baud)
    try:
        time.sleep(0.5)   # native CDC: no auto-reset on open, just let RX settle
        c.clear()

        print("\n1) hello handshake")
        c.send({"cmd": "hello", "host": "smoketest", "proto": 1})
        hello = c.wait_for(lambda m: m.get("type") == "hello", 2.0)
        check("hello reply", hello is not None)
        if hello:
            check("proto 1", hello.get("proto") == 1, f"proto={hello.get('proto')}")
            check("caps has radar (not the drive base)",
                  "radar" in (hello.get("caps") or []), f"caps={hello.get('caps')}")
            sensors = hello.get("sensors") or []
            check("sensor table present", len(sensors) >= 1,
                  f"{len(sensors)} sensors, mounts={[s.get('mount') for s in sensors]}")

        print("\n2) telemetry stream")
        c.clear()
        tel = c.wait_for(lambda m: m.get("type") == "telemetry", 2.0)
        check("telemetry within 2s", tel is not None)
        if tel:
            radar = tel.get("radar") or {}
            check("radar block schema",
                  all(k in radar for k in ("ok", "up", "targets")),
                  f"ok={radar.get('ok')} up={radar.get('up')}")
            check("per-sensor health list", len(tel.get("sens") or []) >= 1,
                  json.dumps(tel.get("sens")))
            # ~10 Hz: expect >=8 frames over 1.2 s
            c.clear()
            time.sleep(1.2)
            with c._lock:
                n = sum(1 for m in c.messages if m.get("type") == "telemetry")
            check("rate ~10 Hz", n >= 8, f"{n} frames / 1.2 s")

        print("\n3) target sanity")
        deadline = time.time() + (args.seconds if args.expect_targets else 2.0)
        best: list[dict] = []
        bad_field = None
        while time.time() < deadline:
            m = c.wait_for(lambda m: m.get("type") == "telemetry"
                           and (m.get("radar") or {}).get("targets"), 0.5)
            if not m:
                continue
            best = (m.get("radar") or {}).get("targets") or []
            for t in best:
                if not (-180.0 < float(t.get("b", 999)) <= 180.0):
                    bad_field = f"bearing {t.get('b')}"
                if not (0.1 <= float(t.get("r", -1)) <= 8.5):
                    bad_field = f"range {t.get('r')}"
                if not (0.0 <= float(t.get("c", -1)) <= 1.0):
                    bad_field = f"confidence {t.get('c')}"
            if best:
                break
        if args.expect_targets:
            check("saw a fused target", bool(best),
                  " | ".join(f"{t['b']}° {t['r']}m c={t['c']}" for t in best) or "none")
        if best:
            check("target fields in range", bad_field is None, bad_field or "")
        elif not args.expect_targets:
            print("  (no targets in view — field checks skipped; use --expect-targets to require one)")

        print("\n4) error handling")
        c.clear()
        c.send({"cmd": "warp_drive", "seq": 9001})
        ack = c.wait_for(lambda m: m.get("type") == "ack" and m.get("seq") == 9001, 2.0)
        check("unknown cmd acked", ack is not None and ack.get("accepted") is False
              and ack.get("reason") == "unknown_cmd", json.dumps(ack))
        c.ser.write(b'{"v":99,"cmd":"hello","seq":9002}\n')
        ack = c.wait_for(lambda m: m.get("type") == "ack" and m.get("seq") == 9002, 2.0)
        check("bad version acked", ack is not None and ack.get("accepted") is False
              and ack.get("reason") == "bad_version", json.dumps(ack))

        print("\n5) link health")
        check("no parse errors", c.parse_errors == 0, f"parse_errors={c.parse_errors}")
    finally:
        passed = sum(1 for _, ok, _ in RESULTS if ok)
        total = len(RESULTS)
        print(f"\n=== {passed}/{total} checks passed; parse_errors={c.parse_errors} ===")
        c.close()
    return 0 if passed == total else 1


if __name__ == "__main__":
    sys.exit(main())
