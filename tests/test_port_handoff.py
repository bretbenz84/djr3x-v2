"""Tests for utils/port_handoff.py — the main.py ↔ menu bar serial handoff.

No serial port is opened anywhere here (opening one reboots the Arduinos), and
lsof is stubbed rather than run, so these are pure logic tests.
"""

import os
import subprocess
import unittest
from unittest import mock

from utils import port_handoff


class _FakeClock:
    """Monotonic clock that only advances when the code under test sleeps."""

    def __init__(self):
        self.now = 0.0
        self.slept = []

    def sleep(self, secs):
        self.slept.append(secs)
        self.now += secs

    def monotonic(self):
        return self.now


class HoldersParseTest(unittest.TestCase):
    """`lsof -Fpn` field output → {device: [pid, ...]}."""

    DEVICES = ["/dev/cu.usbmodem004830621", "/dev/cu.usbserial-110", "/dev/cu.usbmodem1301"]

    def _run(self, stdout, returncode=1):
        completed = subprocess.CompletedProcess(args=[], returncode=returncode, stdout=stdout, stderr="")
        with mock.patch.object(port_handoff, "_lsof_path", return_value="/usr/sbin/lsof"), \
             mock.patch.object(port_handoff.subprocess, "run", return_value=completed):
            return port_handoff.holders(self.DEVICES)

    def test_parses_pid_per_device(self):
        # Real shape captured from `lsof -Fpn` on the robot Mac. Status 1 means
        # "one of these files has no holder" — the normal case, not an error.
        held = self._run(
            "p742\nf3\nn/dev/cu.usbserial-110\n"
            "p750\nf3\nn/dev/cu.usbmodem004830621\n"
        )
        self.assertEqual(
            held,
            {"/dev/cu.usbserial-110": [742], "/dev/cu.usbmodem004830621": [750]},
        )
        self.assertNotIn("/dev/cu.usbmodem1301", held)

    def test_empty_output_means_nothing_held(self):
        self.assertEqual(self._run(""), {})

    def test_our_own_pid_is_not_a_holder(self):
        held = self._run(f"p{os.getpid()}\nf3\nn/dev/cu.usbserial-110\n")
        self.assertEqual(held, {})

    def test_one_process_holding_two_devices(self):
        held = self._run(
            "p742\nf3\nn/dev/cu.usbserial-110\nf4\nn/dev/cu.usbmodem1301\n"
        )
        self.assertEqual(
            held, {"/dev/cu.usbserial-110": [742], "/dev/cu.usbmodem1301": [742]}
        )

    def test_unexpected_exit_status_is_cant_tell_not_free(self):
        self.assertIsNone(self._run("", returncode=127))

    def test_lsof_missing_is_cant_tell_not_free(self):
        with mock.patch.object(port_handoff, "_lsof_path", return_value=None):
            self.assertIsNone(port_handoff.holders(self.DEVICES))

    def test_lsof_blowing_up_is_cant_tell_not_free(self):
        with mock.patch.object(port_handoff, "_lsof_path", return_value="/usr/sbin/lsof"), \
             mock.patch.object(port_handoff.subprocess, "run", side_effect=OSError("boom")):
            self.assertIsNone(port_handoff.holders(self.DEVICES))

    def test_no_devices_short_circuits_without_running_lsof(self):
        with mock.patch.object(port_handoff.subprocess, "run") as run:
            self.assertEqual(port_handoff.holders([None, ""]), {})
        run.assert_not_called()


class WaitForReleaseTest(unittest.TestCase):
    PORTS = [("Maestro servos", "/dev/cu.maestro"), ("chest LEDs", "/dev/cu.chest")]

    def _wait(self, holder_results, **kw):
        """Run wait_for_release with a scripted sequence of holders() answers."""
        clock = _FakeClock()
        with mock.patch.object(port_handoff, "holders", side_effect=holder_results) as holders, \
             mock.patch.object(port_handoff, "describe_holder", side_effect=lambda p: f"fake (pid {p})"):
            stuck = port_handoff.wait_for_release(
                self.PORTS, sleep=clock.sleep, monotonic=clock.monotonic, **kw
            )
        return stuck, clock, holders

    def test_free_ports_return_immediately_without_sleeping(self):
        stuck, clock, holders = self._wait([{}])
        self.assertEqual(stuck, [])
        self.assertEqual(clock.slept, [])       # startup pays nothing in the common case
        self.assertEqual(holders.call_count, 1)  # exactly one lsof

    def test_unconfigured_ports_are_skipped_entirely(self):
        with mock.patch.object(port_handoff, "holders") as holders:
            self.assertEqual(port_handoff.wait_for_release([("motion base", None)]), [])
        holders.assert_not_called()

    def test_zero_timeout_is_a_clean_opt_out(self):
        with mock.patch.object(port_handoff, "holders") as holders:
            self.assertEqual(port_handoff.wait_for_release(self.PORTS, timeout=0), [])
        holders.assert_not_called()

    def test_waits_until_the_companion_lets_go(self):
        stuck, clock, holders = self._wait(
            [{"/dev/cu.maestro": [750]}, {"/dev/cu.maestro": [750]}, {}],
            timeout=5.0,
            poll=0.15,
        )
        self.assertEqual(stuck, [])
        self.assertEqual(holders.call_count, 3)
        self.assertAlmostEqual(clock.now, 0.30, places=6)

    def test_timeout_reports_the_stuck_port_and_gives_up(self):
        held = {"/dev/cu.chest": [740]}
        stuck, clock, _ = self._wait([held] * 200, timeout=1.0, poll=0.15)
        self.assertEqual(stuck, ["chest LEDs"])
        # Bounded: we never block startup past the configured ceiling.
        self.assertLessEqual(clock.now, 1.0 + 0.15)

    def test_only_the_still_held_port_is_named(self):
        held = {"/dev/cu.chest": [740]}
        stuck, _, _ = self._wait([held] * 200, timeout=0.3, poll=0.15)
        self.assertEqual(stuck, ["chest LEDs"])
        self.assertNotIn("Maestro servos", stuck)

    def test_no_lsof_falls_back_to_one_blind_grace_pause(self):
        stuck, clock, holders = self._wait([None], blind_grace=1.5)
        self.assertEqual(stuck, [])
        self.assertEqual(clock.slept, [1.5])     # sleep past one companion poll
        self.assertEqual(holders.call_count, 1)  # and do not spin on it

    def test_lsof_vanishing_mid_wait_does_not_hang(self):
        stuck, _, _ = self._wait([{"/dev/cu.maestro": [750]}, None], timeout=5.0, poll=0.15)
        self.assertEqual(stuck, [])


class MainWiringTest(unittest.TestCase):
    """The handoff must be unmissable and unable to break startup."""

    def test_hardware_init_waits_before_the_first_connect(self):
        source = (_project_root() / "main.py").read_text()
        handoff = source.index("_wait_for_companion_port_handoff()\n\n    servo_ok")
        self.assertGreater(handoff, source.index("=== Initializing hardware ==="))

    def test_handoff_failure_cannot_prevent_startup(self):
        source = (_project_root() / "main.py").read_text()
        body = source[source.index("def _wait_for_companion_port_handoff"):]
        body = body[: body.index("\ndef ")]
        self.assertIn("except Exception", body)

    def test_supervisor_marks_its_own_launches(self):
        source = (_project_root() / "rex_supervisor.py").read_text()
        self.assertIn('DJR3X_LAUNCHED_BY="supervisor"', source)


def _project_root():
    from pathlib import Path
    return Path(__file__).resolve().parent.parent


if __name__ == "__main__":
    unittest.main()
