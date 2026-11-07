"""Deterministic disk scanning — the `du`/`find` layer, in Python.

We don't scan the whole disk blindly. We look where space actually hides:
  - cache/support containers (each child is a candidate),
  - project roots (find node_modules/.venv/build dirs, without descending into
    them),
  - a handful of known heavy singletons.

Only items at or above `min_size` are returned, so the report stays readable.

A ScanProgress object counts files and bytes as we go (for the live display)
and lets the caller stop the walk (Ctrl+C).

Folders are sized one at a time by default. Measured on a warm disk cache,
threads made sizing 3-4x *slower*: the per-file work is Python code, and
Python runs it on one core however many threads wait for it. On a cold cache
(the first scan after a restart) the disk may be the bottleneck instead, and
threads could help there; set DISKSAGE_SCAN_WORKERS=4 to measure it.
"""

from __future__ import annotations

import os
import stat
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field

HOME = os.path.expanduser("~")


def _workers_from_env() -> int:
    try:
        return max(1, min(32, int(os.environ.get("DISKSAGE_SCAN_WORKERS", "1"))))
    except ValueError:
        return 1


# How many folders to size at once (see the module docstring).
SCAN_WORKERS = _workers_from_env()

# macOS <sys/stat.h>: the file's contents live in iCloud / a File Provider, not on
# this disk. Python 3.13+ exposes it as stat.SF_DATALESS.
SF_DATALESS = getattr(stat, "SF_DATALESS", 0x40000000)

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
    # ML leftovers: often big, but they can hold results you can't regenerate,
    # so they're always REVIEW and never pre-ticked (see knowledge.py).
    "wandb", "mlruns", "lightning_logs", ".ipynb_checkpoints",
}

# Files that change when you work on a project. Their newest mtime is a good
# stand-in for "when did I last work on this".
PROJECT_MARKERS = (
    "package.json", "pyproject.toml", "requirements.txt", "setup.py", "Cargo.toml",
    "go.mod", "pom.xml", "build.gradle", "Gemfile", "composer.json",
    ".git/index", ".git/HEAD",
)

# Known heavy singletons worth surfacing directly.
SINGLETONS = [
    os.path.join(HOME, "Library/Arduino15/staging"),
    os.path.join(HOME, "Library/Arduino15/packages"),
    os.path.join(HOME, "Library/Developer/Xcode/DerivedData"),
    os.path.join(HOME, "Library/Developer/Xcode/iOS DeviceSupport"),
    # (~/.Trash is deliberately not here: it can't be moved into itself, and
    #  `disksage clean` offers to empty it at the end instead.)
    os.path.join(HOME, ".m2/repository"),
    os.path.join(HOME, ".gradle/caches"),
    # local model stores — big, outside the usual cache/support dirs
    os.path.join(HOME, ".lmstudio/models"),
    os.path.join(HOME, ".ollama/models"),
    os.path.join(HOME, ".keras"),
]

# Folders that hold several separately-reclaimable things. Instead of offering
# the whole folder, each child is its own candidate — e.g. Hugging Face keeps
# models, datasets, a download cache and your login token side by side.
NESTED_CONTAINERS = [
    os.path.join(HOME, ".cache/huggingface"),
]

# Directories never descended into when hunting for big loose files.
FILE_WALK_SKIP = CRUFT_DIR_NAMES | {".git"}

# macOS packages: folders Finder shows as a single file. Their insides are an
# app's data (a Photos library, an app bundle, a Final Cut library, a VM), so we
# never descend into them and the safety floor never lets anything inside go.
PACKAGE_SUFFIXES = (
    ".app", ".bundle", ".framework", ".plugin", ".kext",
    ".photoslibrary", ".photolibrary", ".aplibrary", ".migratedphotolibrary",
    ".fcpbundle", ".imovielibrary", ".logicx", ".band", ".musiclibrary", ".tvlibrary",
    ".lrlibrary", ".xcarchive", ".xcodeproj", ".xcworkspace", ".playground",
    ".pages", ".numbers", ".key", ".rtfd", ".scriv",
    ".sparsebundle", ".pvm", ".vmwarevm", ".utm",
)


def is_package(name_or_path: str) -> bool:
    return name_or_path.rstrip(os.sep).lower().endswith(PACKAGE_SUFFIXES)


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
    # filled in by the CLI for project folders
    tracked: int = 0           # files under this path that git tracks (source, not output)
    touched: float = 0.0       # when the project around it was last worked on (Unix time)
    # duplicates only: the identical copy that stays
    original: str = ""


class ScanProgress:
    """Counts files looked at and bytes counted while a scan runs, for the live
    display, and lets the caller stop all workers at once.

    Workers count locally and add their totals every FLUSH files, so keeping
    count costs almost nothing even across millions of files."""

    FLUSH = 2000

    def __init__(self) -> None:
        self.files = 0
        self.bytes = 0
        self.started = time.monotonic()
        self._lock = threading.Lock()
        self._stop = threading.Event()

    def add(self, files: int, nbytes: int) -> None:
        if files or nbytes:
            with self._lock:
                self.files += files
                self.bytes += nbytes

    def cancel(self) -> None:
        self._stop.set()

    @property
    def cancelled(self) -> bool:
        return self._stop.is_set()

    @property
    def seconds(self) -> float:
        return time.monotonic() - self.started


def size_on_disk(st: os.stat_result) -> int:
    """Bytes a file really occupies.

    st_size is the file's *length*. Sparse files (VM images), cloud placeholders
    and compressed files take far less space on disk, so totals built from
    st_size promise space that deleting will never free."""
    blocks = getattr(st, "st_blocks", None)
    return blocks * 512 if blocks is not None else st.st_size


def is_dataless(st: os.stat_result) -> bool:
    """True for iCloud / Dropbox / OneDrive placeholders whose contents aren't on
    this Mac. Deleting one frees nothing locally, and reading one (to hash it)
    makes macOS download it."""
    return bool(getattr(st, "st_flags", 0) & SF_DATALESS)


# Several folders are sized at once, and a hard-linked file can appear in two
# of them; this lock keeps "have we counted this inode yet?" exact.
_seen_lock = threading.Lock()


def _count_once(st: os.stat_result, seen: set[tuple[int, int]]) -> int:
    """Bytes on disk for one file, counting a hard-linked file only the first
    time we meet it (the same bytes can have several names)."""
    if st.st_nlink > 1:
        key = (st.st_dev, st.st_ino)
        with _seen_lock:
            if key in seen:
                return 0
            seen.add(key)
    return size_on_disk(st)


def dir_size(path: str, stop: set[str] | None = None,
             seen: set[tuple[int, int]] | None = None,
             progress: ScanProgress | None = None) -> int:
    """Bytes on disk under path, not following symlinks.

    `stop` is a set of paths NOT to descend into — used so a parent folder
    doesn't double-count bytes that belong to a separately-listed child
    (e.g. Application Support/Claude excludes its own vm_bundles). `path`
    itself may be in it.

    `seen` is shared across one whole scan so a hard-linked file is only
    counted for the first candidate that reaches it.

    `progress` gets running counts, and stops the walk early if cancelled."""
    stop = stop or set()
    seen = set() if seen is None else seen
    try:
        st = os.lstat(path)
    except OSError:
        return 0
    if stat.S_ISLNK(st.st_mode):
        return 0
    if not stat.S_ISDIR(st.st_mode):
        size = _count_once(st, seen)
        if progress:
            progress.add(1, size)
        return size
    total = 0
    pending_files = pending_bytes = 0
    stack = [path]
    while stack:
        if progress and progress.cancelled:
            break
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
                            size = _count_once(entry.stat(follow_symlinks=False), seen)
                            total += size
                            pending_files += 1
                            pending_bytes += size
                            if progress and pending_files >= ScanProgress.FLUSH:
                                progress.add(pending_files, pending_bytes)
                                pending_files = pending_bytes = 0
                    except OSError:
                        continue
        except OSError:
            continue
    if progress:
        progress.add(pending_files, pending_bytes)
    return total


def _once_each(paths, follow_symlinks: bool) -> list[str]:
    """`paths` without the ones that are another name for an earlier one.

    On a case-insensitive disk (the macOS default) ~/Projects and ~/projects
    are the same folder, and a symlink can alias another root. Walking both
    listed every project folder twice and counted its size twice.

    Roots are compared following symlinks (an alias of a root *is* that root);
    candidates are compared without following them, so a symlink is never
    mistaken for the folder it points to."""
    seen: set[tuple[int, int]] = set()
    out = []
    for p in paths:
        try:
            st = os.stat(p) if follow_symlinks else os.lstat(p)
        except OSError:
            continue
        key = (st.st_dev, st.st_ino)
        if key not in seen:
            seen.add(key)
            out.append(p)
    return out


def project_last_touched(cruft_dir: str) -> float:
    """When the project around a cruft folder (its parent) was last worked on,
    as a Unix timestamp; 0.0 if unknown."""
    project = os.path.dirname(os.path.realpath(cruft_dir).rstrip(os.sep))
    times = []
    for marker in PROJECT_MARKERS:
        try:
            times.append(os.stat(os.path.join(project, marker)).st_mtime)
        except OSError:
            pass
    if not times:
        try:
            times.append(os.stat(project).st_mtime)
        except OSError:
            return 0.0
    return max(times)


@dataclass(frozen=True)
class FileInfo:
    path: str
    size: int      # length in bytes: identical files have equal lengths, so group duplicates by this
    on_disk: int   # bytes allocated: what deleting the file frees; show and total this


@dataclass
class ProjectWalk:
    """What one pass over the project folders found."""
    cruft: list[str] = field(default_factory=list)       # node_modules, .venv, build, …
    files: list[FileInfo] = field(default_factory=list)  # big loose files, if asked for
    cloud_only_files: int = 0
    cloud_only_bytes: int = 0


def _map(fn, items: list, progress: ScanProgress | None = None) -> list:
    """`[fn(x) for x in items]`, on SCAN_WORKERS threads when that's more than
    one. Ctrl+C cancels the other workers instead of waiting for them."""
    if SCAN_WORKERS <= 1 or len(items) < 2:
        return [fn(x) for x in items]
    with ThreadPoolExecutor(max_workers=SCAN_WORKERS) as pool:
        try:
            return list(pool.map(fn, items))
        except BaseException:
            if progress:
                progress.cancel()
            raise


def _walk_project_root(root: str, file_min: int | None, seen: set[tuple[int, int]],
                       progress: ScanProgress | None) -> ProjectWalk:
    found = ProjectWalk()
    base_depth = root.rstrip(os.sep).count(os.sep)
    pending = 0
    for dirpath, dirnames, filenames in os.walk(root):
        if progress and progress.cancelled:
            break
        if dirpath.count(os.sep) - base_depth >= MAX_WALK_DEPTH:
            dirnames[:] = []
            continue
        descend = []
        for name in dirnames:
            if is_package(name) or name == ".git":
                continue  # an app's data, or git's internals: never ours to judge
            if name in CRUFT_DIR_NAMES:
                found.cruft.append(os.path.join(dirpath, name))
                continue  # listed (and sized) as a folder of its own
            descend.append(name)
        dirnames[:] = descend
        if file_min is None:
            continue
        for name in filenames:
            path = os.path.join(dirpath, name)
            try:
                st = os.lstat(path)
            except OSError:
                continue
            pending += 1
            if progress and pending >= ScanProgress.FLUSH:
                progress.add(pending, 0)
                pending = 0
            if not stat.S_ISREG(st.st_mode) or st.st_size < file_min:
                continue  # symlinks, sockets, small files
            if is_dataless(st):
                found.cloud_only_files += 1
                found.cloud_only_bytes += st.st_size
                continue
            key = (st.st_dev, st.st_ino)
            with _seen_lock:
                if key in seen:
                    continue  # another name for a file we already have
                seen.add(key)
            found.files.append(FileInfo(path, st.st_size, size_on_disk(st)))
    if progress:
        progress.add(pending, 0)
    return found


def walk_projects(roots: list[str], file_min: int | None = None,
                  progress: ScanProgress | None = None) -> ProjectWalk:
    """One pass over the project folders that finds both the cruft folders
    (without descending into them) and, when `file_min` is given, the big loose
    files. Each root is walked in its own thread.

    Never returns the same bytes twice: symlinks are skipped, and only the first
    name of a hard-linked file is kept, so a file can't be offered as a
    "duplicate" of itself. Cloud placeholders are skipped and counted (they free
    nothing, and reading them downloads them)."""
    roots = [r for r in _once_each(roots, follow_symlinks=True) if os.path.isdir(r)]
    seen: set[tuple[int, int]] = set()
    parts = _map(lambda r: _walk_project_root(r, file_min, seen, progress), roots, progress)
    out = ProjectWalk()
    for part in parts:
        out.cruft.extend(part.cruft)
        out.files.extend(part.files)
        out.cloud_only_files += part.cloud_only_files
        out.cloud_only_bytes += part.cloud_only_bytes
    return out


def _drill_one(top: str, base_depth: int) -> list[str]:
    """Buried caches inside one app's folder in Application Support."""
    name = os.path.basename(top)
    if is_package(name):
        return []
    if name in BURIED_CACHE_NAMES:
        return [top]
    found = []
    for dirpath, dirnames, _ in os.walk(top):
        if dirpath.count(os.sep) - base_depth >= DRILL_DEPTH:
            dirnames[:] = []
            continue
        for d in list(dirnames):
            if is_package(d):
                dirnames.remove(d)
            elif d in BURIED_CACHE_NAMES:
                found.append(os.path.join(dirpath, d))
                dirnames.remove(d)  # don't descend further into it
    return found


def _drill_for_buried() -> list[str]:
    tops: list[tuple[str, int]] = []
    for root in DRILL_ROOTS:
        try:
            names = os.listdir(root)
        except OSError:
            continue
        depth = root.rstrip(os.sep).count(os.sep)
        tops.extend((os.path.join(root, n), depth) for n in names
                    if os.path.isdir(os.path.join(root, n)))
    return [p for part in _map(lambda t: _drill_one(*t), tops) for p in part]


def collect_candidates(root: str | None = None, cruft: list[str] | None = None) -> list[str]:
    """Every folder worth sizing. Pass `cruft` from an earlier walk_projects()
    call to avoid walking the project folders a second time."""
    if cruft is None:
        cruft = walk_projects([root] if root else PROJECT_ROOTS).cruft

    # Custom-root mode: just this folder's cruft dirs (node_modules, .venv,
    # build, …). Big loose files and duplicates inside it are surfaced
    # separately by the file scan, so we deliberately do NOT list every
    # subfolder as a blob — that would double-count the files within them.
    if root:
        return _once_each(sorted(set(cruft)), follow_symlinks=False)

    paths: set[str] = set()

    for container in CONTAINER_DIRS:
        if os.path.isdir(container):
            try:
                for name in os.listdir(container):
                    paths.add(os.path.join(container, name))
            except OSError:
                pass

    for nested in NESTED_CONTAINERS:
        if nested in paths and os.path.isdir(nested):
            paths.discard(nested)
            try:
                for name in os.listdir(nested):
                    paths.add(os.path.join(nested, name))
            except OSError:
                pass

    paths.update(cruft)
    paths.update(_drill_for_buried())

    for s in SINGLETONS:
        if os.path.exists(s):
            paths.add(s)

    return _once_each(sorted(paths), follow_symlinks=False)


def scan_dirs(min_size: int = 20 * 1024 * 1024, root: str | None = None,
              cruft: list[str] | None = None,
              progress: ScanProgress | None = None) -> list[Item]:
    """Return directory candidates >= min_size, largest first.

    Sizes are NON-OVERLAPPING: when one candidate lives inside another, the
    parent's size excludes the child, so the same bytes are never counted
    twice and every folder shows its own reclaimable weight. Sizes are bytes
    on disk, and a hard-linked file counts once across the whole scan."""
    candidates = collect_candidates(root, cruft=cruft)
    stop = set(candidates)  # no folder descends into another candidate
    seen: set[tuple[int, int]] = set()

    sizes = _map(lambda p: dir_size(p, stop=stop, seen=seen, progress=progress),
                 candidates, progress)

    items = [Item(path=p, size=s, kind="dir")
             for p, s in zip(candidates, sizes, strict=True) if s >= min_size]
    items.sort(key=lambda i: i.size, reverse=True)
    return items


def collect_files(file_min: int, roots: list[str] | None = None,
                  stats: dict | None = None) -> list[FileInfo]:
    """All regular files >= file_min bytes inside the given roots (default:
    project roots), skipping cruft/.git dirs (already accounted for as
    directories). See walk_projects(); pass a `stats` dict to get how many
    cloud-only files were skipped."""
    walk = walk_projects(roots if roots is not None else PROJECT_ROOTS, file_min=file_min)
    if stats is not None:
        stats["cloud_only_files"] = stats.get("cloud_only_files", 0) + walk.cloud_only_files
        stats["cloud_only_bytes"] = stats.get("cloud_only_bytes", 0) + walk.cloud_only_bytes
    return walk.files


# Backwards-compatible alias.
def scan(min_size: int = 20 * 1024 * 1024) -> list[Item]:
    return scan_dirs(min_size)
