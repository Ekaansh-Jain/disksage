"""Other tools' own cleanup commands.

Some space belongs to a tool that knows exactly what's still in use. Its own
command can free space without throwing away what you still need (uv, pnpm,
Homebrew, Hugging Face keep what your projects use), and for Docker it's the
only option: Docker's disk image lives in ~/Library/Containers, which disksage
never scans or touches.

disksage only shows these as suggestions in `scan`. The one it offers to run
in `clean` is Docker's, after showing Docker's own report and asking first.
"""

from __future__ import annotations

import shutil
import subprocess
from dataclasses import dataclass


@dataclass(frozen=True)
class NativeCleanup:
    tool: str                 # executable that must be on PATH
    command: tuple[str, ...]
    removes: str              # shown to the user before running

    @property
    def display(self) -> str:
        return " ".join(self.command)


DOCKER_PRUNE = NativeCleanup(
    "docker", ("docker", "system", "prune", "--force"),
    "stopped containers, unused networks, dangling images and build cache")

NATIVE_CLEANUPS = (
    DOCKER_PRUNE,
    NativeCleanup("brew", ("brew", "cleanup", "--prune=all"),
                  "old versions of installed formulae and every cached download"),
    NativeCleanup("xcrun", ("xcrun", "simctl", "delete", "unavailable"),
                  "simulators for Xcode versions you no longer have"),
    NativeCleanup("uv", ("uv", "cache", "prune"),
                  "uv cache entries no environment uses"),
    NativeCleanup("pnpm", ("pnpm", "store", "prune"),
                  "packages no project references"),
    NativeCleanup("conda", ("conda", "clean", "--all", "--yes"),
                  "conda package tarballs and caches"),
    # Newer huggingface_hub versions share one file between repos, so don't
    # delete single blobs by hand: `hf cache rm <repo>` removes a model,
    # `hf cache prune` removes files no cached model uses anymore.
    NativeCleanup("hf", ("hf", "cache", "prune"),
                  "Hugging Face files no cached model uses anymore"),
)


def available() -> list[NativeCleanup]:
    """The cleanups whose tool is installed."""
    return [c for c in NATIVE_CLEANUPS if shutil.which(c.tool)]


def docker_usage(timeout: float = 15) -> str | None:
    """Docker's own space report (`docker system df`), or None when Docker isn't
    installed or isn't running."""
    if not shutil.which("docker"):
        return None
    try:
        result = subprocess.run(["docker", "system", "df"],
                                capture_output=True, text=True, timeout=timeout)
    except (OSError, subprocess.SubprocessError):
        return None
    return result.stdout.strip() if result.returncode == 0 and result.stdout.strip() else None


def run(cleanup: NativeCleanup, timeout: float = 1800) -> tuple[bool, str]:
    """Run one cleanup. Returns (succeeded, last line of its output)."""
    try:
        result = subprocess.run(list(cleanup.command), capture_output=True, text=True,
                                timeout=timeout)
    except (OSError, subprocess.SubprocessError) as exc:
        return False, str(exc)
    lines = (result.stdout or result.stderr or "").strip().splitlines()
    return result.returncode == 0, lines[-1] if lines else ""
