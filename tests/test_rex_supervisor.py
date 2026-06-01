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


class SupervisorInputScalingTest(unittest.TestCase):
    """openWakeWord needs int16-range PCM. Feeding the raw float32 [-1,1] that
    sounddevice returns pins every score at ~0.001 and the wake word NEVER fires
    (the real "nothing happened" bug). _to_oww_input must rescale to int16."""

    def setUp(self):
        self.sup = _load_supervisor()

    def test_scales_float_to_int16_range(self):
        import numpy as np
        out = self.sup._to_oww_input(np.array([0.0, 1.0, -1.0, 0.5], dtype=np.float32))
        self.assertEqual(out.dtype, np.int16)
        self.assertEqual(int(out[0]), 0)
        self.assertEqual(int(out[1]), 32767)
        self.assertEqual(int(out[2]), -32767)
        # A loud full-scale signal must map to int16 magnitudes, not stay ~±1.
        self.assertGreater(int(np.max(np.abs(out))), 30000)

    def test_clips_out_of_range_input(self):
        import numpy as np
        out = self.sup._to_oww_input(np.array([2.0, -3.0], dtype=np.float32))
        self.assertEqual(int(out[0]), 32767)
        self.assertEqual(int(out[1]), -32767)

    def test_no_transcription_machinery_remains(self):
        # The helper is ONNX-only now: no VAD / Whisper / RMS-gate / phrase matcher.
        for dead in ("_chunk_has_speech", "_load_vad", "_transcribe",
                     "_transcript_is_wake_phrase", "_RMS_GATE", "_WAKE_MODE"):
            self.assertFalse(hasattr(self.sup, dead),
                             f"{dead} should be gone in the ONNX-only supervisor")

    def test_silent_input_scores_below_threshold(self):
        # End-to-end through the real model: silence must not false-trigger.
        import numpy as np
        model = self.sup._load_model()
        self.assertIsNotNone(model)
        scores = model.predict(self.sup._to_oww_input(np.zeros(self.sup._CHUNK_SAMPLES, dtype=np.float32)))
        self.assertLess(max(scores.values()), self.sup._wake_threshold())


class SupervisorEnvParsingTest(unittest.TestCase):
    """The .env mic device must resolve correctly (the cause of 'no trigger')."""

    def setUp(self):
        self.sup = _load_supervisor()

    def test_env_parser_strips_surrounding_quotes(self):
        import tempfile
        from pathlib import Path
        with tempfile.TemporaryDirectory() as d:
            envfile = Path(d) / ".env"
            envfile.write_text(
                'AUDIO_DEVICE_NAME="MacBook Pro Microphone"\n'
                "AUDIO_DEVICE_INDEX=\n"
                "OTHER='single quoted'\n"
                "PLAIN=bare\n"
            )
            with mock.patch.object(self.sup, "_PROJECT_ROOT", Path(d)):
                env = self.sup._read_env_file()
        # Quotes must be gone so the name matches a real device.
        self.assertEqual(env["AUDIO_DEVICE_NAME"], "MacBook Pro Microphone")
        self.assertEqual(env["OTHER"], "single quoted")
        self.assertEqual(env["PLAIN"], "bare")

    def _clear_audio_env(self):
        # _resolve_input_device reads os.environ first; clear so the passed-in
        # dict is the only source (the dev box / shell may export these).
        return mock.patch.dict(
            os.environ, {"AUDIO_DEVICE_NAME": "", "AUDIO_DEVICE_INDEX": ""}, clear=False
        )

    def test_resolve_input_device_matches_case_insensitively(self):
        # Mock the device list so the test is hardware-independent.
        with self._clear_audio_env(), mock.patch.object(
            self.sup, "_list_input_devices",
            return_value=[(0, "ReSpeaker Lite"), (1, "MacBook Pro Microphone")],
        ):
            idx = self.sup._resolve_input_device({"AUDIO_DEVICE_NAME": "macbook pro microphone"})
        self.assertEqual(idx, 1)

    def test_resolve_input_device_substring_match(self):
        with self._clear_audio_env(), mock.patch.object(
            self.sup, "_list_input_devices",
            return_value=[(0, "ReSpeaker Lite"), (2, "MacBook Air Microphone")],
        ):
            idx = self.sup._resolve_input_device({"AUDIO_DEVICE_NAME": "ReSpeaker"})
        self.assertEqual(idx, 0)

    def test_resolve_falls_back_to_index_when_name_absent(self):
        with self._clear_audio_env(), mock.patch.object(
            self.sup, "_list_input_devices", return_value=[(0, "X")]
        ):
            idx = self.sup._resolve_input_device({"AUDIO_DEVICE_INDEX": "3"})
        self.assertEqual(idx, 3)


class SupervisorChimeTest(unittest.TestCase):
    def setUp(self):
        self.sup = _load_supervisor()

    def test_chime_file_exists(self):
        self.assertTrue(self.sup._CHIME_FILE.exists(), "startup chime asset missing")

    def test_chime_uses_afplay_when_available(self):
        import shutil
        with (
            mock.patch.object(self.sup, "_CHIME_ENABLED", True),
            mock.patch.object(shutil, "which", return_value="/usr/bin/afplay"),
            mock.patch.object(self.sup.subprocess, "Popen") as popen,
        ):
            self.sup._play_chime()
        popen.assert_called_once()
        args = popen.call_args.args[0]
        self.assertEqual(args[0], "/usr/bin/afplay")
        self.assertEqual(args[1], str(self.sup._CHIME_FILE))

    def test_chime_disabled_plays_nothing(self):
        with (
            mock.patch.object(self.sup, "_CHIME_ENABLED", False),
            mock.patch.object(self.sup.subprocess, "Popen") as popen,
        ):
            self.sup._play_chime()
        popen.assert_not_called()

    def test_chime_missing_file_warns_no_crash(self):
        from pathlib import Path
        with (
            mock.patch.object(self.sup, "_CHIME_ENABLED", True),
            mock.patch.object(self.sup, "_CHIME_FILE", Path("/nonexistent/chime.mp3")),
            mock.patch.object(self.sup.subprocess, "Popen") as popen,
        ):
            self.sup._play_chime()  # must not raise
        popen.assert_not_called()


if __name__ == "__main__":
    unittest.main()
