"""The safety floor.

This layer sits UNDER the AI. No matter how confident the model is, nothing
here lets it delete something dangerous. The rules are dumb, deterministic, and
unbreakable on purpose:

  1. Only ever act inside the user's home directory.
  2. Never touch a protected path (keychains, sandbox data, top-level folders).
  3. Warn before removing a git repo that has unpushed or uncommitted work.
  4. Delete = move to Trash (send2trash), never `rm -rf`. Mistakes are undoable.
"""

from __future__ import annotations

import os
import subprocess
from fnmatch import fnmatch

HOME = os.path.expanduser("~")

# Paths that must never be removed, even if the AI tags them "safe".
PROTECTED_GLOBS = [
    os.path.join(HOME, "Library/Keychains*"),
    os.path.join(HOME, "Library/Containers*"),
    os.path.join(HOME, "Library/Group Containers*"),
    os.path.join(HOME, "Library/Mail*"),
    os.path.join(HOME, "Library/Messages*"),
    os.path.join(HOME, "Library/Preferences*"),
    "*.photoslibrary",
]

# Top-level home folders whose *own* directory must never be deleted
# (we still allow deleting regenerable children inside them, e.g. node_modules).
PROTECTED_ROOTS = {
    os.path.join(HOME, name)
    for name in (
        "Documents", "Desktop", "Downloads", "Pictures", "Movies",
        "Music", "Library", "Applications", "Public", "Developer",
    )
}


def _within_home(path: str) -> bool:
    real = os.path.realpath(path)
    return real == HOME or real.startswith(HOME + os.sep)


def is_safe_to_trash(path: str) -> tuple[bool, str]:
    """Deterministic veto. Returns (allowed, reason_if_blocked)."""
    real = os.path.realpath(path)

    if not os.path.exists(path):
        return False, "path no longer exists"
    if not _within_home(path):
        return False, "outside your home directory"
    if real == HOME:
        return False, "this is your home directory"
    if real in PROTECTED_ROOTS:
        return False, "protected top-level folder"
    for glob in PROTECTED_GLOBS:
        if fnmatch(real, glob) or fnmatch(path, glob):
            return False, "protected system/app data"
    return True, ""


def git_risk(path: str) -> str | None:
    """If path is (or contains) a git repo with work not on a remote, describe
    the risk. Returns None when there's nothing to lose."""
    git_dir = path if os.path.basename(path) == ".git" else os.path.join(path, ".git")
    repo = os.path.dirname(git_dir) if os.path.basename(path) == ".git" else path
    if not os.path.isdir(git_dir):
        return None
    # Never let git block on a credential/terminal prompt, and don't take locks.
    env = {**os.environ, "GIT_TERMINAL_PROMPT": "0", "GIT_OPTIONAL_LOCKS": "0"}
    try:
        unpushed = subprocess.run(
            ["git", "-C", repo, "log", "--branches", "--not", "--remotes", "--oneline"],
            capture_output=True, text=True, timeout=8, env=env,
        ).stdout.strip()
        dirty = subprocess.run(
            ["git", "-C", repo, "status", "--short", "--untracked-files=no"],
            capture_output=True, text=True, timeout=8, env=env,
        ).stdout.strip()
    except Exception:
        return "could not verify git state — check manually before deleting"

    parts = []
    if unpushed:
        parts.append(f"{len(unpushed.splitlines())} commit(s) not pushed to any remote")
    if dirty:
        parts.append(f"{len(dirty.splitlines())} uncommitted change(s)")
    return "; ".join(parts) if parts else None


def trash(path: str) -> None:
    """Reversible delete. Raises on failure so the caller can report it."""
    from send2trash import send2trash
    send2trash(path)
