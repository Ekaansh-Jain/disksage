"""Deterministic disk scanning — the `du`/`find` layer, in Python.

We don't scan the whole disk blindly. We look where space actually hides:
  - cache/support containers (each child is a candidate),
  - project roots (find node_modules/.venv/build dirs, without descending into
    them),
  - a handful of known heavy singletons.

Only items at or above `min_size` are returned, so the report stays readable.
"""

from __future__ import annotations

import os
from dataclasses import dataclass

HOME = os.path.expanduser("~")

# Each immediate child of these is a candidate.
CONTAINER_DIRS = [
    os.path.join(HOME, "Library/Caches"),
    os.path.join(HOME, "Library/Application Support"),
    os.path.join(HOME, "Library/Logs"),
    os.path.join(HOME, ".cache"),
]

# Where developer cruft accumulates. We walk these to find the dirs below.
PROJECT_ROOTS = [
    os.path.join(HOME, d)
    for d in ("Desktop", "Documents", "Developer", "Projects", "projects",
              "code", "Code", "src", "git", "repos", "work")
]

# Directory names that are reclaimable wherever they appear in a project.
CRUFT_DIR_NAMES = {
    "node_modules", ".venv", "venv", "__pycache__", ".pytest_cache",
    ".mypy_cache", ".ruff_cache", ".turbo", "target", "dist", "build",
    ".next", ".nuxt", ".gradle",
}

# Known heavy singletons worth surfacing directly.
SINGLETONS = [
    os.path.join(HOME, "Library/Arduino15/staging"),
    os.path.join(HOME, "Library/Arduino15/packages"),
    os.path.join(HOME, "Library/Developer/Xcode/DerivedData"),
    os.path.join(HOME, "Library/Developer/Xcode/iOS DeviceSupport"),
    os.path.join(HOME, ".Trash"),
    os.path.join(HOME, ".m2/repository"),
    os.path.join(HOME, ".gradle/caches"),
    # local model stores — big, outside the usual cache/support dirs
    os.path.join(HOME, ".lmstudio/models"),
    os.path.join(HOME, ".ollama/models"),
    os.path.join(HOME, ".keras"),
]

# Directories never descended into when hunting for big loose files.
FILE_WALK_SKIP = CRUFT_DIR_NAMES | {".git"}

MAX_WALK_DEPTH = 6  # don't spelunk forever inside project roots

# Reclaimable sub-folders that hide INSIDE otherwise-keep app folders
# (Application Support/<app>/...). We drill in and surface these separately so
# a big mixed folder isn't reported as one opaque "keep" blob.
BURIED_CACHE_NAMES = {
    "vm_bundles",            # Claude sandbox images
    "Service Worker",        # browser/electron per-site cache
    "Code Cache", "GPUCache", "ShaderCache", "GrShaderCache",
    "Cache", "Cache_Data", "CachedData", "Crashpad",
    "Component Crx Cache", "component_crx_cache",
}
# Where to look for them, and how deep (Chrome nests them under profiles).
DRILL_ROOTS = [os.path.join(HOME, "Library/Application Support")]
DRILL_DEPTH = 5


@dataclass
class Item:
    path: str
    size: int
    kind: str = "dir"          # "dir" | "file" | "duplicate"
    # filled in later by classify.py
    tier: str = "review"
    label: str = ""
    why: str = ""
    source: str = ""


def dir_size(path: str, stop: set[str] | None = None) -> int:
    """Total bytes under path, not following symlinks.

    `stop` is a set of paths NOT to descend into — used so a parent folder
    doesn't double-count bytes that belong to a separately-listed child
    (e.g. Application Support/Claude excludes its own vm_bundles)."""
    stop = stop or set()
    try:
        if os.path.islink(path):
            return 0
        if os.path.isfile(path):
            return os.path.getsize(path)
    except OSError:
        return 0
    total = 0
    stack = [path]
    while stack:
        d = stack.pop()
        try:
            with os.scandir(d) as it:
                for entry in it:
                    try:
                        if entry.is_symlink():
                            continue
                        if entry.is_dir(follow_symlinks=False):
                            if entry.path in stop:
                                continue  # belongs to a deeper candidate
                            stack.append(entry.path)
                        else:
                            total += entry.stat(follow_symlinks=False).st_size
                    except OSError:
                        continue
        except OSError:
            continue
    return total


def _walk_for_cruft(root: str, found: list[str]) -> None:
    base_depth = root.rstrip(os.sep).count(os.sep)
    for dirpath, dirnames, _ in os.walk(root):
        if dirpath.count(os.sep) - base_depth >= MAX_WALK_DEPTH:
            dirnames[:] = []
            continue
        pruned = []
        for name in list(dirnames):
            full = os.path.join(dirpath, name)
            if name in CRUFT_DIR_NAMES:
                found.append(full)
                pruned.append(name)  # don't descend into a matched dir
        for name in pruned:
            dirnames.remove(name)


def _drill_for_buried(root: str, found: list[str]) -> None:
    base_depth = root.rstrip(os.sep).count(os.sep)
    for dirpath, dirnames, _ in os.walk(root):
        if dirpath.count(os.sep) - base_depth >= DRILL_DEPTH:
            dirnames[:] = []
            continue
        for name in list(dirnames):
            if name in BURIED_CACHE_NAMES:
                found.append(os.path.join(dirpath, name))
                dirnames.remove(name)  # don't descend further into it


def collect_candidates(root: str | None = None) -> list[str]:
    # Custom-root mode: scan just this folder for cruft dirs (node_modules,
    # .venv, build, …). Big loose files and duplicates inside it are surfaced
    # separately by the file scan, so we deliberately do NOT list every
    # subfolder as a blob — that would double-count the files within them.
    if root:
        cruft: list[str] = []
        _walk_for_cruft(root, cruft)
        return sorted(set(cruft))

    paths: set[str] = set()

    for container in CONTAINER_DIRS:
        if os.path.isdir(container):
            try:
                for name in os.listdir(container):
                    paths.add(os.path.join(container, name))
            except OSError:
                pass

    cruft: list[str] = []
    for root in PROJECT_ROOTS:
        if os.path.isdir(root):
            _walk_for_cruft(root, cruft)
    paths.update(cruft)

    buried: list[str] = []
    for root in DRILL_ROOTS:
        if os.path.isdir(root):
            _drill_for_buried(root, buried)
    paths.update(buried)

    for s in SINGLETONS:
        if os.path.exists(s):
            paths.add(s)

    return sorted(paths)


def scan_dirs(min_size: int = 20 * 1024 * 1024, root: str | None = None) -> list[Item]:
    """Return directory candidates >= min_size, largest first.

    Sizes are NON-OVERLAPPING: when one candidate lives inside another, the
    parent's size excludes the child, so the same bytes are never counted
    twice and every folder shows its own reclaimable weight."""
    candidates = collect_candidates(root)
    cand_set = set(candidates)

    items: list[Item] = []
    for path in candidates:
        others = cand_set - {path}
        size = dir_size(path, stop=others)
        if size >= min_size:
            items.append(Item(path=path, size=size, kind="dir"))
    items.sort(key=lambda i: i.size, reverse=True)
    return items


def collect_files(file_min: int, roots: list[str] | None = None) -> list[tuple[str, int]]:
    """All files >= file_min bytes inside the given roots (default: project
    roots), skipping cruft/.git dirs (already accounted for as directories)."""
    out: list[tuple[str, int]] = []
    for root in (roots if roots is not None else PROJECT_ROOTS):
        if not os.path.isdir(root):
            continue
        base_depth = root.rstrip(os.sep).count(os.sep)
        for dirpath, dirnames, filenames in os.walk(root):
            if dirpath.count(os.sep) - base_depth >= MAX_WALK_DEPTH:
                dirnames[:] = []
                continue
            dirnames[:] = [d for d in dirnames if d not in FILE_WALK_SKIP]
            for name in filenames:
                fp = os.path.join(dirpath, name)
                try:
                    if os.path.islink(fp):
                        continue
                    size = os.path.getsize(fp)
                except OSError:
                    continue
                if size >= file_min:
                    out.append((fp, size))
    return out


# Backwards-compatible alias.
def scan(min_size: int = 20 * 1024 * 1024) -> list[Item]:
    return scan_dirs(min_size)
