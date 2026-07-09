"""Safe, stateless Git updates for the robot runtime.

The physical robot deploys from ``origin/main``.  This module fetches that ref and
fast-forwards the checked-out ``main`` branch only when the worktree is clean and
the caller says it is safe to change files.  It deliberately never stashes,
resets, merges divergent history, or writes updater state files.
"""

from __future__ import annotations

import logging
import os
import shutil
import subprocess
from dataclasses import dataclass
from pathlib import Path


log = logging.getLogger(__name__)


@dataclass(frozen=True)
class UpdateResult:
    """Outcome of one fetch/update attempt."""

    checked: bool
    updated: bool = False
    update_available: bool = False
    old_commit: str = ""
    new_commit: str = ""
    reason: str = ""


def auto_update_enabled() -> bool:
    return os.environ.get("REX_AUTO_UPDATE_ENABLED", "1").strip().lower() not in {
        "0", "false", "no", "off",
    }


def update_interval_secs() -> float:
    """Periodic fetch interval; clamped to one minute to prevent tight loops."""
    try:
        return max(60.0, float(os.environ.get("REX_AUTO_UPDATE_INTERVAL_SECS", "14400")))
    except ValueError:
        return 14400.0


def _timeout_secs() -> float:
    try:
        return max(5.0, float(os.environ.get("REX_AUTO_UPDATE_TIMEOUT_SECS", "45")))
    except ValueError:
        return 45.0


def _run_git(project_root: Path, *args: str) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        ["git", *args],
        cwd=str(project_root),
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        timeout=_timeout_secs(),
        check=False,
    )


def _short(commit: str) -> str:
    return commit[:12] if commit else "unknown"


def _failure_message(proc: subprocess.CompletedProcess[str]) -> str:
    return (proc.stderr or proc.stdout or f"git exited {proc.returncode}").strip()


def update_repository(
    project_root: Path,
    *,
    apply: bool,
    trigger: str,
) -> UpdateResult:
    """Fetch ``origin/main`` and optionally fast-forward the local ``main``.

    ``apply=False`` is safe while the controller is running: it updates only
    Git's remote-tracking metadata and reports whether a deployment is waiting.
    All errors fail open so callers can continue running the installed version.
    """
    root = Path(project_root).resolve()
    if not auto_update_enabled():
        return UpdateResult(False, reason="automatic updates disabled")
    if shutil.which("git") is None:
        log.warning("[auto_update] %s: git is unavailable; using installed code.", trigger)
        return UpdateResult(False, reason="git unavailable")
    if not (root / ".git").exists():
        log.warning("[auto_update] %s: %s is not a Git checkout; using installed code.", trigger, root)
        return UpdateResult(False, reason="not a git checkout")

    try:
        branch = _run_git(root, "branch", "--show-current")
        if branch.returncode != 0:
            raise RuntimeError(_failure_message(branch))
        if branch.stdout.strip() != "main":
            reason = f"checked out branch is {branch.stdout.strip() or 'detached'}, not main"
            log.warning("[auto_update] %s: %s; update skipped.", trigger, reason)
            return UpdateResult(False, reason=reason)

        status = _run_git(root, "status", "--porcelain", "--untracked-files=normal")
        if status.returncode != 0:
            raise RuntimeError(_failure_message(status))
        if status.stdout.strip():
            log.warning("[auto_update] %s: worktree is not clean; update skipped.", trigger)
            return UpdateResult(False, reason="worktree is not clean")

        head = _run_git(root, "rev-parse", "HEAD")
        if head.returncode != 0:
            raise RuntimeError(_failure_message(head))
        old_commit = head.stdout.strip()

        log.info("[auto_update] %s: checking origin/main...", trigger)
        fetch = _run_git(root, "fetch", "--quiet", "origin", "main")
        if fetch.returncode != 0:
            raise RuntimeError(_failure_message(fetch))

        remote = _run_git(root, "rev-parse", "origin/main")
        if remote.returncode != 0:
            raise RuntimeError(_failure_message(remote))
        remote_commit = remote.stdout.strip()
        if remote_commit == old_commit:
            log.info("[auto_update] %s: already current at %s.", trigger, _short(old_commit))
            return UpdateResult(True, old_commit=old_commit, new_commit=old_commit, reason="current")

        local_is_ancestor = _run_git(
            root, "merge-base", "--is-ancestor", old_commit, remote_commit,
        )
        if local_is_ancestor.returncode != 0:
            remote_is_ancestor = _run_git(
                root, "merge-base", "--is-ancestor", remote_commit, old_commit,
            )
            if remote_is_ancestor.returncode == 0:
                reason = "local main is ahead of origin/main"
                log.warning("[auto_update] %s: %s; update skipped.", trigger, reason)
                return UpdateResult(
                    True, old_commit=old_commit, new_commit=remote_commit, reason=reason,
                )
            reason = "local main has diverged from origin/main"
            log.error("[auto_update] %s: %s; refusing to merge or reset.", trigger, reason)
            return UpdateResult(
                True, update_available=True, old_commit=old_commit,
                new_commit=remote_commit, reason=reason,
            )

        if not apply:
            log.info(
                "[auto_update] %s: update %s -> %s is waiting; controller is running.",
                trigger, _short(old_commit), _short(remote_commit),
            )
            return UpdateResult(
                True, update_available=True, old_commit=old_commit,
                new_commit=remote_commit, reason="update waiting",
            )

        merge = _run_git(root, "merge", "--ff-only", "origin/main")
        if merge.returncode != 0:
            raise RuntimeError(_failure_message(merge))
        new_head = _run_git(root, "rev-parse", "HEAD")
        if new_head.returncode != 0:
            raise RuntimeError(_failure_message(new_head))
        new_commit = new_head.stdout.strip()
        log.info(
            "[auto_update] %s: updated %s -> %s.",
            trigger, _short(old_commit), _short(new_commit),
        )
        return UpdateResult(
            True, updated=True, old_commit=old_commit,
            new_commit=new_commit, reason="fast-forwarded",
        )
    except subprocess.TimeoutExpired:
        log.warning(
            "[auto_update] %s: Git check timed out; using installed code.", trigger,
        )
        return UpdateResult(False, reason="git timeout")
    except Exception as exc:
        log.warning(
            "[auto_update] %s: update check failed (%s); using installed code.",
            trigger, exc,
        )
        return UpdateResult(False, reason=str(exc))
