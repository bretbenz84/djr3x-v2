import os
import subprocess
import tempfile
import unittest
from pathlib import Path
from unittest import mock

from utils import repo_updater


def _git(cwd: Path, *args: str) -> str:
    proc = subprocess.run(
        ["git", *args], cwd=str(cwd), text=True,
        stdout=subprocess.PIPE, stderr=subprocess.PIPE, check=False,
    )
    if proc.returncode != 0:
        raise AssertionError(f"git {' '.join(args)} failed: {proc.stderr}")
    return proc.stdout.strip()


class RepoUpdaterTest(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        root = Path(self.tmp.name)
        self.origin = root / "origin.git"
        self.author = root / "author"
        self.robot = root / "robot"

        _git(root, "init", "--bare", str(self.origin))
        _git(root, "clone", str(self.origin), str(self.author))
        _git(self.author, "config", "user.name", "Test Author")
        _git(self.author, "config", "user.email", "test@example.com")
        _git(self.author, "checkout", "-b", "main")
        (self.author / "version.txt").write_text("one\n")
        _git(self.author, "add", "version.txt")
        _git(self.author, "commit", "-m", "initial")
        _git(self.author, "push", "-u", "origin", "main")
        _git(root, "clone", "--branch", "main", str(self.origin), str(self.robot))

        self.env = mock.patch.dict(
            os.environ,
            {
                "REX_AUTO_UPDATE_ENABLED": "1",
                "REX_AUTO_UPDATE_TIMEOUT_SECS": "10",
            },
            clear=False,
        )
        self.env.start()

    def tearDown(self):
        self.env.stop()
        self.tmp.cleanup()

    def _push_update(self):
        (self.author / "version.txt").write_text("two\n")
        _git(self.author, "add", "version.txt")
        _git(self.author, "commit", "-m", "second")
        _git(self.author, "push", "origin", "main")

    def test_fast_forwards_clean_main(self):
        old = _git(self.robot, "rev-parse", "HEAD")
        self._push_update()

        result = repo_updater.update_repository(
            self.robot, apply=True, trigger="test launch",
        )

        self.assertTrue(result.checked)
        self.assertTrue(result.updated)
        self.assertNotEqual(result.new_commit, old)
        self.assertEqual((self.robot / "version.txt").read_text(), "two\n")

    def test_check_only_fetches_but_does_not_change_worktree(self):
        old = _git(self.robot, "rev-parse", "HEAD")
        self._push_update()

        result = repo_updater.update_repository(
            self.robot, apply=False, trigger="controller running",
        )

        self.assertTrue(result.checked)
        self.assertFalse(result.updated)
        self.assertTrue(result.update_available)
        self.assertEqual(_git(self.robot, "rev-parse", "HEAD"), old)
        self.assertNotEqual(_git(self.robot, "rev-parse", "origin/main"), old)
        self.assertEqual((self.robot / "version.txt").read_text(), "one\n")

    def test_dirty_worktree_is_never_changed(self):
        self._push_update()
        (self.robot / "version.txt").write_text("local robot edit\n")
        old = _git(self.robot, "rev-parse", "HEAD")

        result = repo_updater.update_repository(
            self.robot, apply=True, trigger="dirty test",
        )

        self.assertFalse(result.checked)
        self.assertFalse(result.updated)
        self.assertIn("not clean", result.reason)
        self.assertEqual(_git(self.robot, "rev-parse", "HEAD"), old)
        self.assertEqual((self.robot / "version.txt").read_text(), "local robot edit\n")

    def test_non_main_branch_is_never_changed(self):
        _git(self.robot, "checkout", "-b", "experiment")
        old = _git(self.robot, "rev-parse", "HEAD")

        result = repo_updater.update_repository(
            self.robot, apply=True, trigger="branch test",
        )

        self.assertFalse(result.checked)
        self.assertFalse(result.updated)
        self.assertIn("not main", result.reason)
        self.assertEqual(_git(self.robot, "rev-parse", "HEAD"), old)

    def test_local_main_ahead_is_not_reset(self):
        _git(self.robot, "config", "user.name", "Robot Test")
        _git(self.robot, "config", "user.email", "robot@example.com")
        (self.robot / "robot-only.txt").write_text("local commit\n")
        _git(self.robot, "add", "robot-only.txt")
        _git(self.robot, "commit", "-m", "local robot commit")
        old = _git(self.robot, "rev-parse", "HEAD")

        result = repo_updater.update_repository(
            self.robot, apply=True, trigger="ahead test",
        )

        self.assertTrue(result.checked)
        self.assertFalse(result.updated)
        self.assertFalse(result.update_available)
        self.assertIn("ahead", result.reason)
        self.assertEqual(_git(self.robot, "rev-parse", "HEAD"), old)

    def test_disabled_does_not_invoke_git(self):
        with (
            mock.patch.dict(os.environ, {"REX_AUTO_UPDATE_ENABLED": "0"}),
            mock.patch.object(repo_updater, "_run_git") as run_git,
        ):
            result = repo_updater.update_repository(
                self.robot, apply=True, trigger="disabled test",
            )
        self.assertFalse(result.checked)
        self.assertEqual(result.reason, "automatic updates disabled")
        run_git.assert_not_called()

    def test_interval_defaults_to_four_hours_and_is_clamped(self):
        with mock.patch.dict(os.environ, {}, clear=False):
            os.environ.pop("REX_AUTO_UPDATE_INTERVAL_SECS", None)
            self.assertEqual(repo_updater.update_interval_secs(), 14400.0)
        with mock.patch.dict(os.environ, {"REX_AUTO_UPDATE_INTERVAL_SECS": "5"}):
            self.assertEqual(repo_updater.update_interval_secs(), 60.0)


if __name__ == "__main__":
    unittest.main()
