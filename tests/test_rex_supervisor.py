import importlib.util
import os
import subprocess
import sys
import tempfile
import time
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


class SupervisorLivenessTest(unittest.TestCase):
    """The supervisor must stay dormant whenever a controller is alive (awake or
    asleep) so it never spawns a second main.py."""

    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self._lock = Path(self._tmp.name) / "djr3x-main.lock"
        self._old_env = os.environ.get("DJR3X_LOCK_PATH")
        os.environ["DJR3X_LOCK_PATH"] = str(self._lock)
        from utils import single_instance
        single_instance.release()
        self.si = single_instance
        self.sup = _load_supervisor()

    def tearDown(self):
        self.si.release()
        if self._old_env is None:
            os.environ.pop("DJR3X_LOCK_PATH", None)
        else:
            os.environ["DJR3X_LOCK_PATH"] = self._old_env
        self._tmp.cleanup()

    def test_dormant_when_lock_held_by_another_process(self):
        # Simulate a running/sleeping controller: another process holds the lock.
        code = (
            "import os,sys;"
            "sys.path.insert(0, os.environ['DJR3X_REPO']);"
            "from utils import single_instance as s;"
            "s.acquire();"
            "sys.stdout.write('held\\n');sys.stdout.flush();"
            "sys.stdin.readline()"
        )
        env = dict(os.environ)
        env["DJR3X_REPO"] = str(_REPO)
        proc = subprocess.Popen(
            [sys.executable, "-c", code],
            stdin=subprocess.PIPE, stdout=subprocess.PIPE, text=True, env=env,
        )
        try:
            self.assertEqual(proc.stdout.readline().strip(), "held")
            # No child of ours, but the lock is held → controller IS running.
            self.assertTrue(self.sup._controller_running(None))
        finally:
            proc.stdin.write("go\n"); proc.stdin.flush()
            proc.wait(timeout=5)
            proc.stdin.close(); proc.stdout.close()

    def test_active_when_no_controller_and_no_child(self):
        self.assertFalse(self.sup._controller_running(None))

    def test_running_when_own_child_is_alive(self):
        fake_child = mock.Mock()
        fake_child.poll.return_value = None  # still running
        self.assertTrue(self.sup._controller_running(fake_child))

    def test_not_running_when_child_exited_and_lock_free(self):
        fake_child = mock.Mock()
        fake_child.poll.return_value = 0  # exited
        self.assertFalse(self.sup._controller_running(fake_child))


class SupervisorModelTest(unittest.TestCase):
    def test_wakeuprex_model_loads_and_predicts(self):
        sup = _load_supervisor()
        model = sup._load_model()
        self.assertIsNotNone(model, "wakeuprex model failed to load")
        import numpy as np
        scores = model.predict(np.zeros(sup._CHUNK_SAMPLES, dtype=np.float32))
        self.assertIn("wakeuprex", scores)

    def test_threshold_env_override(self):
        sup = _load_supervisor()
        with mock.patch.dict(os.environ, {"REX_SUPERVISOR_WAKE_THRESHOLD": "0.7"}):
            self.assertAlmostEqual(sup._wake_threshold(), 0.7)
        with mock.patch.dict(os.environ, {}, clear=False):
            os.environ.pop("REX_SUPERVISOR_WAKE_THRESHOLD", None)
            self.assertAlmostEqual(sup._wake_threshold(), 0.5)


class SupervisorWakePhraseTest(unittest.TestCase):
    """The transcription wake path matches the same phrases the main app accepts
    from SLEEP (intelligence.interaction._is_sleep_wake_transcript)."""

    def setUp(self):
        self.sup = _load_supervisor()

    def test_accepts_wake_phrases(self):
        for text in (
            "wake up rex",
            "wake up rex.",
            "Wake up, Rex!",
            "hey wake up rex",
            "please wake up rex please",
            "rex wake up",
            "dj rex wake up",
            "wake up r3x",
            "wakeuprex",
        ):
            with self.subTest(text=text):
                self.assertTrue(self.sup._transcript_is_wake_phrase(text))

    def test_rejects_non_wake_phrases(self):
        for text in (
            "",
            "what's the weather",
            "wake up the kids",
            "i need to wake up early",
            "tell rex something",
            "rex is a good droid",
            "go to sleep",
        ):
            with self.subTest(text=text):
                self.assertFalse(self.sup._transcript_is_wake_phrase(text))

    def test_wake_mode_default_is_both(self):
        # Default mode runs both detectors so the reliable transcription path is on.
        self.assertEqual(self.sup._WAKE_MODE, "both")


if __name__ == "__main__":
    unittest.main()
