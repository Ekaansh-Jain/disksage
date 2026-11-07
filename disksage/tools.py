"""Other tools' own cleanup commands.

Some space belongs to a tool that knows exactly what's still in use. Its own
command can free space without throwing away what you still need (uv, pnpm,
Homebrew, Hugging Face keep what your projects use), and for Docker it's the
only option: Docker's disk image lives in ~/Library/Containers, which disksage
never scans or touches.

disksage only shows these as suggestions in `scan`. The one it can run in
`clean` is Docker's: it appears in the checklist, unticked, with Docker's own
numbers.
"""

from __future__ import annotations

import json
import re
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


@dataclass(frozen=True)
class DockerReport:
    rows: tuple[tuple[str, str, str], ...]   # (type, size, reclaimable) as Docker prints them
    reclaimable: int                         # bytes, added up from Docker's own numbers


# Docker prints decimal units (1 kB = 1000 B).
_UNITS = {"b": 1, "kb": 10**3, "mb": 10**6, "gb": 10**9, "tb": 10**12}
_SIZE = re.compile(r"\s*([\d.]+)\s*([kmgt]?b)", re.IGNORECASE)


def parse_docker_size(text: str) -> int:
    """'3.1GB (73%)' -> 3100000000. Unreadable text counts as 0."""
    match = _SIZE.match(text or "")
    if not match:
        return 0
    try:
        return int(float(match.group(1)) * _UNITS[match.group(2).lower()])
    except (ValueError, KeyError):
        return 0


def docker_report(timeout: float = 5) -> DockerReport | None:
    """Docker's own space report, or None when Docker isn't installed or isn't
    running. The timeout is short because Docker can take a while to answer
    while it starts, and the scan shouldn't wait on it."""
    if not shutil.which("docker"):
        return None
    try:
        result = subprocess.run(["docker", "system", "df", "--format", "{{json .}}"],
                                capture_output=True, text=True, timeout=timeout)
    except (OSError, subprocess.SubprocessError):
        return None
    if result.returncode != 0:
        return None
    rows = []
    for line in result.stdout.splitlines():
        try:
            data = json.loads(line)
        except ValueError:
            continue
        if isinstance(data, dict):
            rows.append((str(data.get("Type", "")), str(data.get("Size", "")),
                         str(data.get("Reclaimable", ""))))
    if not rows:
        return None
    return DockerReport(tuple(rows), sum(parse_docker_size(r[2]) for r in rows))


def run(cleanup: NativeCleanup, timeout: float = 1800) -> tuple[bool, str]:
    """Run one cleanup. Returns (succeeded, last line of its output)."""
    try:
        result = subprocess.run(list(cleanup.command), capture_output=True, text=True,
                                timeout=timeout)
    except (OSError, subprocess.SubprocessError) as exc:
        return False, str(exc)
    lines = (result.stdout or result.stderr or "").strip().splitlines()
    return result.returncode == 0, lines[-1] if lines else ""
