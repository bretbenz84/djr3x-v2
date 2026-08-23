"""Park the servos after a full test sweep — keep this module named to run LAST.

Many test modules exercise gesture/tracking code against the REAL Maestro when
one is connected (the hardware layer is deliberately not mocked everywhere), so
a full per-module sweep leaves the head wherever the last fixture commanded it —
a random pose. The next real boot then hurts twice: `animations.startup()` seeds
its wake glide from SHUTDOWN_REST_POSE (where shutdown()/sleep() parked it), so
from a random test pose the head JERKS to the first commanded target instead of
rising smoothly (field 2026-08-11: "he's started up and jerked to these
locations").

The single test below glides every channel back to the shutdown rest pose (head
lowered, tilt down, visor closed, neck centred, elbow at its unpowered rest, the
other arm channels neutral) — the exact pose startup() assumes. The module name
starts with `test_zzz` so an alphabetical per-module sweep (the standard
full-suite runner, see CLAUDE.md) reaches it last. With no Maestro connected
(CI, a laptop) the test SKIPS and touches nothing.
"""

import time
import unittest

import config
from hardware import servos


class ServoParkTest(unittest.TestCase):
    def test_park_all_servos_at_the_shutdown_rest_pose(self):
        if not servos.connected() and not servos.connect():
            self.skipTest("no Maestro connected — nothing to park")
        from sequences import animations

        # A test may have left breathing/arm-idle threads, the GUI manual
        # override, or the sleep latch armed — all of which would freeze or
        # fight the park.
        try:
            servos.set_manual_override_enabled(False)
        except Exception:
            pass
        try:
            servos.release_sleep_latch()
        except Exception:
            pass
        try:
            servos.stop_breathing()
        except Exception:
            pass
        try:
            servos.pause_arm_idle()
        except Exception:
            pass

        # Clear any slow per-channel speed/accel a fixture left behind, or the
        # physical servo lags the streamed targets and the process exits with
        # the head stranded mid-travel (same lesson as animations.shutdown()).
        servos.set_motion_profile(
            list(config.HEAD_CHANNELS) + list(config.ARM_CHANNELS),
            speed=int(getattr(config, "SHUTDOWN_DROOP_SERVO_SPEED", 70)),
            acceleration=int(getattr(config, "SHUTDOWN_DROOP_SERVO_ACCELERATION", 14)),
        )

        # SHUTDOWN_REST_POSE already carries the elbow at ELBOW_REST (where the limp
        # arm falls once the power is off) — don't put it back to neutral here, or
        # the next cold boot jerks the fallen arm out of it.
        targets = dict(animations.SHUTDOWN_REST_POSE)
        targets.update({
            5: animations.HAND_NEUTRAL,
            6: animations.POKERARM_NEUTRAL,
            7: animations.HEROARM_NEUTRAL,
        })
        servos.move_to(targets, step_us=50, step_delay=0.012)

        # Let the physical travel finish before the interpreter (and the serial
        # link) goes away.
        time.sleep(float(getattr(config, "SHUTDOWN_DROOP_SETTLE_SECS", 0.8)))
        self.assertTrue(servos.connected(), "park move should leave the link up")


if __name__ == "__main__":
    unittest.main()
