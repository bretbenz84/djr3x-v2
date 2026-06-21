#!/usr/bin/env python3
"""
Standalone smoke test for the wave-back ARM gesture (sequences.animations.wave_back_gesture).

This bypasses the whole camera / detection / proactive-speech path and just drives the
servos directly, so you can answer one question on the robot: does the wrist (the "hand"
servo, ch5) physically move when the wave-back gesture runs?

Run on the robot (servos connected):

    venv/bin/python tools/wave_back_smoketest.py            # 4 wrist sweeps (default)
    venv/bin/python tools/wave_back_smoketest.py 6          # 6 sweeps

It prints the resolved servo limits it will sweep between, then runs the gesture
synchronously and reports success/failure. Watch the arm: it should raise, then the wrist
should sweep between both limits N times, then return.
"""

import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))  # project root on sys.path

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(name)-22s | %(levelname)-7s | %(message)s",
)
_log = logging.getLogger("wave_back_smoketest")


def main() -> int:
    count = 4
    if len(sys.argv) > 1:
        try:
            count = int(sys.argv[1])
        except ValueError:
            print(f"usage: {sys.argv[0]} [sweep_count]")
            return 2

    import config
    from hardware import servos
    from sequences import animations

    hand = config.SERVO_CHANNELS.get("hand", {})
    elbow = config.SERVO_CHANNELS.get("elbow", {})
    print(f"hand (wrist, ch{hand.get('ch')}): min={hand.get('min')} max={hand.get('max')} "
          f"neutral={hand.get('neutral')}")
    print(f"elbow (ch{elbow.get('ch')}): min={elbow.get('min')} max={elbow.get('max')} "
          f"neutral={elbow.get('neutral')}")

    ok = servos.connect()
    print(f"servos.connect() -> {ok}; SERVOS_ENABLED={getattr(servos, 'SERVOS_ENABLED', '?')}")
    if not ok:
        print("Maestro not connected — set MAESTRO_PORT in .env and re-run on the robot.")
        return 1

    print(f"Running wave_back_gesture(count={count}) synchronously — watch the arm/wrist ...")
    result = animations.wave_back_gesture(count=count, async_=False)
    print(f"wave_back_gesture returned {result}")
    print("Did the WRIST sweep between both limits? If the limits above look tiny/equal, the "
          "hand-servo travel needs configuring in .env (SERVO_HAND_MIN_US / SERVO_HAND_MAX_US).")
    return 0 if result else 1


if __name__ == "__main__":
    raise SystemExit(main())
