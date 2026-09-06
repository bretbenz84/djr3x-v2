"""The supervisor answers a mic-wedge exit (code 86) with a relaunch.

Field 2026-09-05 17:15: audio/stream.py's watchdog exited main.py because the
ReSpeaker's CoreAudio client was wedged and could not be reopened in-process.
The supervisor logged "Controller exited (code=86). Resuming wake-word
listening." — i.e. treated it like "shut down" — and Rex stayed silent
mid-conversation until someone said the wake phrase again. A wedge is a restart
request, not a goodbye; these pin the policy that turns it into one, bounded so
a device that wedges at every boot can't loop forever.
"""

import importlib.util
import os
import unittest
from pathlib import Path
from unittest import mock

_REPO = Path(__file__).resolve().parent.parent


def _load_supervisor():
    spec = importlib.util.spec_from_file_location(
        "rex_supervisor", _REPO / "rex_supervisor.py"
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class WedgeRelaunchPolicyTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.sup = _load_supervisor()

    def _policy(self, **kw):
        base = dict(enabled=True, max_relaunches=3, window_secs=600.0)
        base.update(kw)
        return self.sup.WedgeRelaunchPolicy(**base)

    def test_exit_code_matches_the_watchdog(self):
        from audio import stream
        self.assertEqual(self.sup.MIC_WEDGE_EXIT_CODE, stream._DEAD_MIC_EXIT_CODE)

    def test_only_the_wedge_code_relaunches(self):
        p = self._policy()
        for code in (0, 1, -9, None, 85, 87):
            self.assertFalse(p.should_relaunch(code, now=100.0), code)
        self.assertEqual(p.relaunches_in_window(100.0), 0,
                         "a non-wedge exit must not consume relaunch budget")
        self.assertTrue(p.should_relaunch(self.sup.MIC_WEDGE_EXIT_CODE, now=100.0))

    def test_relaunch_is_recorded_and_capped_inside_the_window(self):
        p = self._policy(max_relaunches=3, window_secs=600.0)
        code = self.sup.MIC_WEDGE_EXIT_CODE
        self.assertTrue(p.should_relaunch(code, now=0.0))
        self.assertTrue(p.should_relaunch(code, now=10.0))
        self.assertTrue(p.should_relaunch(code, now=20.0))
        self.assertEqual(p.relaunches_in_window(20.0), 3)
        self.assertFalse(p.should_relaunch(code, now=30.0),
                         "a fourth wedge inside the window must fall back to listening")
        self.assertEqual(p.relaunches_in_window(30.0), 3,
                         "a refused relaunch must not be counted")

    def test_budget_comes_back_when_the_window_rolls(self):
        p = self._policy(max_relaunches=1, window_secs=100.0)
        code = self.sup.MIC_WEDGE_EXIT_CODE
        self.assertTrue(p.should_relaunch(code, now=0.0))
        self.assertFalse(p.should_relaunch(code, now=50.0))
        self.assertTrue(p.should_relaunch(code, now=101.0))

    def test_disabled_policy_never_relaunches(self):
        p = self._policy(enabled=False)
        self.assertFalse(p.should_relaunch(self.sup.MIC_WEDGE_EXIT_CODE, now=0.0))

    def test_zero_budget_never_relaunches(self):
        p = self._policy(max_relaunches=0)
        self.assertFalse(p.should_relaunch(self.sup.MIC_WEDGE_EXIT_CODE, now=0.0))


class WedgeRelaunchEnvTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.sup = _load_supervisor()

    def test_defaults_on_with_three_per_ten_minutes(self):
        env = {k: v for k, v in os.environ.items()
               if not k.startswith("REX_SUPERVISOR_WEDGE_RELAUNCH")}
        with mock.patch.dict(os.environ, env, clear=True):
            p = self.sup.WedgeRelaunchPolicy.from_env()
            self.assertTrue(p.enabled)
            self.assertEqual(p.max_relaunches, 3)
            self.assertEqual(p.window_secs, 600.0)
            self.assertEqual(self.sup._wedge_relaunch_settle_secs(), 2.0)

    def test_env_overrides(self):
        with mock.patch.dict(os.environ, {
            "REX_SUPERVISOR_WEDGE_RELAUNCH": "0",
            "REX_SUPERVISOR_WEDGE_RELAUNCH_MAX": "5",
            "REX_SUPERVISOR_WEDGE_RELAUNCH_WINDOW_SECS": "120",
            "REX_SUPERVISOR_WEDGE_RELAUNCH_SETTLE_SECS": "0.5",
        }):
            p = self.sup.WedgeRelaunchPolicy.from_env()
            self.assertFalse(p.enabled)
            self.assertEqual(p.max_relaunches, 5)
            self.assertEqual(p.window_secs, 120.0)
            self.assertEqual(self.sup._wedge_relaunch_settle_secs(), 0.5)

    def test_garbage_env_falls_back_to_defaults(self):
        with mock.patch.dict(os.environ, {
            "REX_SUPERVISOR_WEDGE_RELAUNCH_MAX": "lots",
            "REX_SUPERVISOR_WEDGE_RELAUNCH_WINDOW_SECS": "soon",
            "REX_SUPERVISOR_WEDGE_RELAUNCH_SETTLE_SECS": "a bit",
        }):
            p = self.sup.WedgeRelaunchPolicy.from_env()
            self.assertEqual(p.max_relaunches, 3)
            self.assertEqual(p.window_secs, 600.0)
            self.assertEqual(self.sup._wedge_relaunch_settle_secs(), 2.0)


class LaunchReasonTest(unittest.TestCase):
    def test_relaunch_log_line_names_the_wedge_not_the_wake_word(self):
        sup = _load_supervisor()
        with mock.patch.object(sup.subprocess, "Popen") as popen, \
             mock.patch.object(sup, "_VENV_PYTHON", Path("/usr/bin/true")), \
             mock.patch.object(sup, "_CONTROLLER_CONSOLE_LOG", Path(os.devnull)), \
             self.assertLogs(sup.log, level="INFO") as captured:
            sup._launch_controller(reason="Mic wedge restart")
        popen.assert_called_once()
        joined = "\n".join(captured.output)
        self.assertIn("Mic wedge restart", joined)
        self.assertNotIn("Wake word heard", joined)


if __name__ == "__main__":
    unittest.main()
