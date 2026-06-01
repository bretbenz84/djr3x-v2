import os
import subprocess
import sys
import tempfile
import time
import unittest
from pathlib import Path


class SingleInstanceLockTest(unittest.TestCase):
    def setUp(self):
        # Each test gets a fresh lock path via env override.
        self._tmp = tempfile.TemporaryDirectory()
        self._lock = Path(self._tmp.name) / "djr3x-main.lock"
        self._old_env = os.environ.get("DJR3X_LOCK_PATH")
        os.environ["DJR3X_LOCK_PATH"] = str(self._lock)
        # Import fresh and ensure no stale handle leaks between tests.
        from utils import single_instance
        single_instance.release()
        self.si = single_instance

    def tearDown(self):
        self.si.release()
        if self._old_env is None:
            os.environ.pop("DJR3X_LOCK_PATH", None)
        else:
            os.environ["DJR3X_LOCK_PATH"] = self._old_env
        self._tmp.cleanup()

    def test_lock_path_uses_env_override(self):
        self.assertEqual(self.si.lock_path(), self._lock)

    def test_acquire_succeeds_and_records_pid(self):
        self.assertTrue(self.si.acquire())
        self.assertTrue(self.si.is_held())
        self.assertEqual(self.si.read_owner_pid(), os.getpid())

    def test_acquire_is_idempotent(self):
        self.assertTrue(self.si.acquire())
        self.assertTrue(self.si.acquire())
        self.assertTrue(self.si.is_held())

    def test_release_clears_hold(self):
        self.si.acquire()
        self.si.release()
        self.assertFalse(self.si.is_held())

    def test_is_held_by_other_false_when_we_hold_it(self):
        self.si.acquire()
        # We hold it, so "by other" must be False.
        self.assertFalse(self.si.is_held_by_other())

    def test_is_held_by_other_false_when_unlocked(self):
        self.assertFalse(self.si.is_held_by_other())

    def _spawn_holder(self):
        """Spawn a subprocess that acquires the lock and waits for a stdin line."""
        code = (
            "import os,sys;"
            "sys.path.insert(0, os.environ['DJR3X_REPO']);"
            "from utils import single_instance as s;"
            "ok=s.acquire();"
            "sys.stdout.write(('held' if ok else 'busy')+chr(10));"
            "sys.stdout.flush();"
            "sys.stdin.readline()"  # block until parent says release
        )
        env = dict(os.environ)
        env["DJR3X_REPO"] = str(Path(__file__).resolve().parent.parent)
        proc = subprocess.Popen(
            [sys.executable, "-c", code],
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            text=True,
            env=env,
        )
        first = proc.stdout.readline().strip()
        return proc, first

    def test_is_held_by_other_true_while_another_process_holds_it(self):
        proc, first = self._spawn_holder()
        try:
            self.assertEqual(first, "held")
            # Another live process holds it → True, and we cannot acquire.
            self.assertTrue(self.si.is_held_by_other())
            self.assertFalse(self.si.acquire())
            self.assertFalse(self.si.is_held())
        finally:
            proc.stdin.write("go\n")
            proc.stdin.flush()
            proc.wait(timeout=5)
            proc.stdin.close()
            proc.stdout.close()

    def test_lock_frees_when_holder_process_dies(self):
        proc, first = self._spawn_holder()
        self.assertEqual(first, "held")
        self.assertTrue(self.si.is_held_by_other())
        # Kill the holder — the OS must release the flock automatically.
        proc.kill()
        proc.wait(timeout=5)
        proc.stdin.close()
        proc.stdout.close()
        # Give the OS a beat to reclaim the advisory lock.
        deadline = time.time() + 3.0
        while self.si.is_held_by_other() and time.time() < deadline:
            time.sleep(0.05)
        self.assertFalse(self.si.is_held_by_other())
        # And we can now take it ourselves.
        self.assertTrue(self.si.acquire())


if __name__ == "__main__":
    unittest.main()
