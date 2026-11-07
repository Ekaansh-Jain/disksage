"""Local AI models: find them across apps, and reclaim duplicates as clones.

Ollama, LM Studio and Hugging Face each keep their own copy of a model. Try the
same GGUF in two of them and it's stored twice. Two of them name files by
content hash, which makes duplicates cheap to find:

  - Ollama:        ~/.ollama/models/blobs/sha256-<hex>, manifests say which
                   model uses which blob
  - Hugging Face:  ~/.cache/huggingface/hub/models--org--name/blobs/<sha256>
                   for large files (newer versions may symlink that into a
                   shared store, so we resolve symlinks)
  - LM Studio:     plain .gguf / .safetensors files, hash unknown until read

Reclaiming doesn't delete anything. Each extra copy is replaced by an APFS
clone of one copy: every app keeps a working file at the same path, and the
bytes are stored once (copy-on-write, so editing one never changes another).
"""

from __future__ import annotations

import contextlib
import ctypes
import ctypes.util
import hashlib
import json
import os
import re
import stat
import sys
from dataclasses import dataclass

from .scan import size_on_disk

HOME = os.path.expanduser("~")
_HEX64 = re.compile(r"[0-9a-f]{64}")

# Where `dedupe` remembers the clones it made, so later runs don't report
# already-shared copies as reclaimable again.
STATE_DIR = os.path.join(os.environ.get("XDG_STATE_HOME") or os.path.join(HOME, ".local/state"),
                         "disksage")
CLONES_FILE = os.path.join(STATE_DIR, "clones.json")

STORE_NAMES = {"ollama": "Ollama", "huggingface": "Hugging Face", "lmstudio": "LM Studio"}
REMOVE_HINTS = {
    "ollama": "ollama rm <name>",
    "huggingface": "hf cache rm model/<id>",
    "lmstudio": "LM Studio > My Models",
}


@dataclass
class ModelFile:
    store: str                   # "ollama" | "huggingface" | "lmstudio"
    path: str                    # the real file (symlinks resolved)
    size: int                    # length in bytes
    sha256: str | None           # from the file name when the app names files by hash
    names: tuple[str, ...] = ()  # models that use this file, where known
    on_disk: int = 0             # bytes allocated


def model_dirs(home: str | None = None) -> dict[str, str]:
    """Where each app keeps models, honouring the apps' own environment variables."""
    home = home or HOME
    hf_home = os.environ.get("HF_HOME") or os.path.join(home, ".cache", "huggingface")
    return {
        "ollama": os.environ.get("OLLAMA_MODELS") or os.path.join(home, ".ollama", "models"),
        "huggingface": os.environ.get("HF_HUB_CACHE") or os.path.join(hf_home, "hub"),
        "lmstudio": os.path.join(home, ".lmstudio", "models"),
    }


def _model_file(store, path, digest, names, min_size) -> ModelFile | None:
    try:
        st = os.stat(path)
    except OSError:
        return None
    if not stat.S_ISREG(st.st_mode) or st.st_size < min_size:
        return None
    return ModelFile(store, os.path.realpath(path), st.st_size, digest, tuple(names),
                     size_on_disk(st))


def ollama_files(models_dir: str, min_size: int = 0) -> list[ModelFile]:
    users: dict[str, list[str]] = {}
    manifests = os.path.join(models_dir, "manifests")
    for dirpath, _, filenames in os.walk(manifests):
        for filename in filenames:
            manifest = os.path.join(dirpath, filename)
            try:
                with open(manifest, "rb") as f:
                    layers = json.load(f).get("layers", [])
            except (OSError, ValueError, AttributeError):
                continue
            # manifests/<registry>/<namespace>/<model>/<tag>
            parts = os.path.relpath(manifest, manifests).split(os.sep)
            if len(parts) < 3:
                continue
            model = "/".join(parts[1:-1]).removeprefix("library/")
            for layer in layers if isinstance(layers, list) else []:
                digest = str(layer.get("digest", "")) if isinstance(layer, dict) else ""
                if digest.startswith("sha256:"):
                    users.setdefault(digest[7:], []).append(f"{model}:{parts[-1]}")
    found = []
    blobs = os.path.join(models_dir, "blobs")
    try:
        names = os.listdir(blobs)
    except OSError:
        return found
    for name in names:
        digest = name[7:] if name.startswith("sha256-") else ""
        if _HEX64.fullmatch(digest):
            mf = _model_file("ollama", os.path.join(blobs, name), digest,
                             users.get(digest, ()), min_size)
            if mf:
                found.append(mf)
    return found


def huggingface_files(hub_dir: str, min_size: int = 0) -> list[ModelFile]:
    found = []
    try:
        repos = os.listdir(hub_dir)
    except OSError:
        return found
    for repo in repos:
        if not repo.startswith(("models--", "datasets--")):
            continue
        model = repo.split("--", 1)[1].replace("--", "/")
        blobs = os.path.join(hub_dir, repo, "blobs")
        try:
            names = os.listdir(blobs)
        except OSError:
            continue
        for name in names:
            if _HEX64.fullmatch(name):  # 40-hex names are small git files, not weights
                mf = _model_file("huggingface", os.path.join(blobs, name), name, (model,),
                                 min_size)
                if mf:
                    found.append(mf)
    return found


def lmstudio_files(models_dir: str, min_size: int = 0) -> list[ModelFile]:
    found = []
    for dirpath, _, filenames in os.walk(models_dir):
        for filename in filenames:
            if filename.lower().endswith((".gguf", ".safetensors")):
                model = os.path.relpath(dirpath, models_dir).replace(os.sep, "/")
                mf = _model_file("lmstudio", os.path.join(dirpath, filename), None, (model,),
                                 min_size)
                if mf:
                    found.append(mf)
    return found


def all_model_files(min_size: int = 50 * 1024 * 1024) -> list[ModelFile]:
    """Every model file over min_size, one entry per real file."""
    dirs = model_dirs()
    files = (ollama_files(dirs["ollama"], min_size)
             + huggingface_files(dirs["huggingface"], min_size)
             + lmstudio_files(dirs["lmstudio"], min_size))
    unique: dict[tuple[int, int], ModelFile] = {}
    for f in files:
        try:
            st = os.stat(f.path)
        except OSError:
            continue
        key = (st.st_dev, st.st_ino)
        if key in unique:  # the same file reached twice (symlinks, hard links)
            unique[key].names = tuple(dict.fromkeys(unique[key].names + f.names))
        else:
            unique[key] = f
    return sorted(unique.values(), key=lambda f: -f.size)


def sha256_file(path: str, chunk: int = 8 * 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(chunk), b""):
            digest.update(block)
    return digest.hexdigest()


def find_model_duplicates(files: list[ModelFile]) -> list[list[ModelFile]]:
    """Groups of identical model files.

    Files named by hash match for free. A file with an unknown hash is only read
    when its size equals another file's, so most gigabytes are never hashed.
    Hard links and symlinks to the same file collapse to one entry."""
    by_size: dict[int, list[ModelFile]] = {}
    for f in files:
        by_size.setdefault(f.size, []).append(f)
    by_hash: dict[str, list[ModelFile]] = {}
    for same_size in by_size.values():
        if len(same_size) < 2:
            continue
        for f in same_size:
            if f.sha256 is None:
                try:
                    f.sha256 = sha256_file(f.path)
                except OSError:
                    continue
            by_hash.setdefault(f.sha256, []).append(f)
    groups = []
    for same_hash in by_hash.values():
        unique: dict[tuple[int, int], ModelFile] = {}
        for f in same_hash:
            try:
                st = os.stat(f.path)
            except OSError:
                continue
            unique.setdefault((st.st_dev, st.st_ino), f)
        if len(unique) > 1:
            groups.append(list(unique.values()))
    return groups


_KEEP_ORDER = {"huggingface": 0, "ollama": 1, "lmstudio": 2}


def pick_keeper(group: list[ModelFile]) -> ModelFile:
    """The copy that stays exactly as it is. Hugging Face's copy may be shared
    by several repos, so it's never the one replaced; Ollama's comes next."""
    return min(group, key=lambda f: (_KEEP_ORDER.get(f.store, 9), f.path))


# ---- remembering clones ----

def _inode_key(path: str) -> str | None:
    """device:inode:size:mtime. The inode alone isn't enough: when a file is
    deleted, the filesystem may hand its inode number straight to the next new
    file (ext4 does), and a re-downloaded model would look already shared.
    A re-download also gets a new mtime; `dedupe` keeps the mtime it found."""
    try:
        st = os.stat(path)
    except OSError:
        return None
    return f"{st.st_dev}:{st.st_ino}:{st.st_size}:{st.st_mtime_ns}"


def load_clones() -> dict[str, dict]:
    try:
        with open(CLONES_FILE) as f:
            data = json.load(f)
        return data if isinstance(data, dict) else {}
    except (OSError, ValueError):
        return {}


def save_clones(records: dict[str, dict]) -> None:
    os.makedirs(STATE_DIR, exist_ok=True)
    tmp = CLONES_FILE + ".tmp"
    with open(tmp, "w") as f:
        json.dump(records, f, indent=1)
    os.replace(tmp, CLONES_FILE)


def record_clone(records: dict[str, dict], keep: str, clone: str) -> None:
    key, keeper = _inode_key(clone), _inode_key(keep)
    if key and keeper:
        records[key] = {"keeper": keeper, "path": clone}


def already_shared(records: dict[str, dict], keep: ModelFile, copy: ModelFile) -> bool:
    """True if `copy` is a clone disksage made of this same `keep` file. If either
    file was replaced since (an app re-downloaded it), its inode changed and the
    copy counts as a full duplicate again."""
    record = records.get(_inode_key(copy.path) or "")
    return (bool(record) and record.get("path") == copy.path
            and record.get("keeper") == _inode_key(keep.path))


# ---- reclaiming: macOS + APFS only ----

def _clonefile(src: str, dst: str) -> None:
    """clonefile(2): a copy-on-write clone. Unlike `cp -c`, it fails instead of
    quietly making a full copy when cloning isn't possible."""
    libc = ctypes.CDLL(ctypes.util.find_library("c"), use_errno=True)
    clonefile = libc.clonefile
    clonefile.argtypes = [ctypes.c_char_p, ctypes.c_char_p, ctypes.c_uint32]
    clonefile.restype = ctypes.c_int
    if clonefile(os.fsencode(src), os.fsencode(dst), 0) != 0:
        err = ctypes.get_errno()
        raise OSError(err, os.strerror(err), dst)


def replace_with_clone(keep: str, duplicate: str) -> int:
    """Turn `duplicate` into an APFS clone of `keep`. Both paths keep working and
    the duplicate's own blocks are freed.

    Checks first that both are on one volume, aren't the same file or hard
    links, and have identical contents (by SHA-256, never trusting names). Keeps
    the duplicate's permissions and modification time, and swaps the file in one
    atomic rename. Returns the duplicate's allocated bytes (0 if nothing
    changed); that's an upper bound, so report the measured free-space change."""
    if sys.platform != "darwin":
        return 0
    keep_st, dup_st = os.stat(keep), os.stat(duplicate)
    if keep_st.st_dev != dup_st.st_dev:
        return 0  # clones only work within one volume
    if (keep_st.st_dev, keep_st.st_ino) == (dup_st.st_dev, dup_st.st_ino) or dup_st.st_nlink > 1:
        return 0  # the same file, or other hard links keep the bytes alive anyway
    if keep_st.st_size != dup_st.st_size or sha256_file(keep) != sha256_file(duplicate):
        return 0
    tmp = os.path.join(os.path.dirname(duplicate),
                       f".{os.path.basename(duplicate)}.disksage-clone")
    with contextlib.suppress(FileNotFoundError):
        os.remove(tmp)  # left over from an interrupted run
    _clonefile(keep, tmp)
    try:
        os.chmod(tmp, stat.S_IMODE(dup_st.st_mode))
        os.utime(tmp, ns=(dup_st.st_atime_ns, dup_st.st_mtime_ns))
        os.replace(tmp, duplicate)
    except BaseException:
        with contextlib.suppress(OSError):
            os.remove(tmp)
        raise
    return size_on_disk(dup_st)
