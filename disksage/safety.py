"""The safety floor.

This layer sits UNDER the AI. No matter how confident the model is, nothing
here lets it delete something dangerous. The rules are dumb, deterministic, and
unbreakable on purpose:

  1. Only ever act inside the user's home directory.
  2. Never touch a protected path (keychains, sandbox data, top-level folders,
     cloud-synced folders, credentials) or anything inside a macOS package.
  3. Warn before removing a git repo that has unpushed or uncommitted work,
     or a folder whose files git tracks (that's source, not build output).
  4. Only caches that rebuild themselves (SAFE in the built-in list, never an
     AI label) are deleted outright: putting them in the Trash would free
     nothing until it's emptied. Everything that could be yours moves to the
     Trash (send2trash), so mistakes stay undoable.
"""

from __future__ import annotations

import os
import shutil
import stat
import subprocess
import sys
from fnmatch import fnmatch

from .scan import is_package

# Resolved, because every check below compares resolved paths (and so do the
# protected paths built from it). If HOME sits behind a symlink (macOS temp
# folders live under /var -> /private/var), an unresolved HOME made every path
# look "outside your home", so nothing could be cleaned at all.
HOME = os.path.realpath(os.path.expanduser("~"))

# Paths that must never be removed, even if the AI tags them "safe".
PROTECTED_GLOBS = [
    os.path.join(HOME, "Library/Keychains*"),
    os.path.join(HOME, "Library/Containers*"),
    os.path.join(HOME, "Library/Group Containers*"),
    os.path.join(HOME, "Library/Mail*"),
    os.path.join(HOME, "Library/Messages*"),
    os.path.join(HOME, "Library/Preferences*"),
    # Synced folders: deleting here deletes the file in the cloud and on your
    # other devices, and often frees nothing (the file may be a placeholder).
    os.path.join(HOME, "Library/Mobile Documents*"),         # iCloud Drive
    os.path.join(HOME, "Library/CloudStorage*"),             # Dropbox, OneDrive, Google Drive
    # iPhone/iPad backups: real data. Manage them in Finder instead.
    os.path.join(HOME, "Library/Application Support/MobileSync*"),
    # Credentials and tool config.
    os.path.join(HOME, ".ssh*"),
    os.path.join(HOME, ".gnupg*"),
    os.path.join(HOME, ".aws*"),
    os.path.join(HOME, ".kube*"),
    os.path.join(HOME, ".config*"),
    os.path.join(HOME, ".cache/huggingface/token"),
    os.path.join(HOME, ".cache/huggingface/stored_tokens"),
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


def inside_package(path: str) -> bool:
    """True if any folder above `path` is a macOS package. A file inside a
    Photos library, an app or a Final Cut library is that app's data."""
    parents = os.path.realpath(path).split(os.sep)[:-1]
    return any(part and is_package(part) for part in parents)


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
    if is_package(real) or inside_package(real):
        return False, "inside a macOS package (library, app or bundle)"
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


def tracked_file_count(path: str, timeout: float = 8.0) -> int:
    """How many files under `path` the surrounding git repo tracks.

    git_risk() only notices a .git *inside* a folder. A committed build/ or
    dist/ folder has none, yet its files are source, not output — so ask the
    repo around it. Returns 0 outside a repo or when git isn't installed."""
    real = os.path.realpath(path).rstrip(os.sep)
    parent, name = os.path.split(real)
    env = {**os.environ, "GIT_TERMINAL_PROMPT": "0", "GIT_OPTIONAL_LOCKS": "0"}
    try:
        result = subprocess.run(["git", "-C", parent, "ls-files", "-z", "--", name],
                                capture_output=True, timeout=timeout, env=env)
    except (OSError, subprocess.SubprocessError):
        return 0
    if result.returncode != 0:  # not inside a repo
        return 0
    return result.stdout.count(b"\0")


def trash(path: str) -> None:
    """Reversible delete. Raises on failure so the caller can report it."""
    from send2trash import send2trash
    send2trash(path)


def remove_regenerable(path: str) -> None:
    """Permanently delete a cache that rebuilds itself.

    Only for items whose SAFE tier came from the built-in list (never from the
    AI), and only after is_safe_to_trash() said yes. Raises on failure so the
    caller can report it."""
    st = os.lstat(path)
    if stat.S_ISDIR(st.st_mode):
        shutil.rmtree(path)
    else:
        os.remove(path)  # a file, or a symlink (removes the link, not its target)


def empty_trash(timeout: float = 900) -> tuple[bool, str]:
    """macOS only: empty the Trash through Finder, like Finder > Empty Trash.
    The first time, macOS asks whether your terminal may control Finder.
    Returns (succeeded, error message)."""
    if sys.platform != "darwin":
        return False, "emptying the Trash is only supported on macOS"
    try:
        result = subprocess.run(
            ["osascript", "-e", 'tell application "Finder" to empty trash'],
            capture_output=True, text=True, timeout=timeout)
    except (OSError, subprocess.SubprocessError) as exc:
        return False, str(exc)
    return result.returncode == 0, result.stderr.strip()
