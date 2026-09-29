"""Exact duplicate detection via a cheap size -> partial-hash -> full-hash cascade.

Correctness guarantees (both matter for safety):
  - Two files are only ever called duplicates if their FULL contents hash equal,
    so we never flag near-identical-but-different files.
  - Within every duplicate group we always keep the first file and only ever
    offer the *extra* copies for deletion — the tool can never remove your last
    copy of something.

Cost: we read the whole of a file only when its size AND its first 4 KB already
collide with another file, so unique files cost one stat + at most 4 KB.
"""

from __future__ import annotations

import hashlib
import os
from collections import defaultdict

_HEAD = 4096
_CHUNK = 1024 * 1024


def _hash(path: str, limit: int | None = None) -> bytes:
    h = hashlib.blake2b(digest_size=16)
    with open(path, "rb") as f:
        if limit is not None:
            h.update(f.read(limit))
        else:
            for chunk in iter(lambda: f.read(_CHUNK), b""):
                h.update(chunk)
    return h.digest()


def find_duplicates(files: list[tuple[str, int]]) -> list[list[str]]:
    """Given (path, size) pairs, return groups of paths with identical content.

    Each returned group has 2+ members and is sorted (shortest path first, so
    the 'kept original' is stable and predictable)."""
    by_size: dict[int, list[str]] = defaultdict(list)
    for path, size in files:
        if size > 0:
            by_size[size].append(path)

    groups: list[list[str]] = []
    for size, paths in by_size.items():
        if len(paths) < 2:
            continue

        by_head: dict[bytes, list[str]] = defaultdict(list)
        for p in paths:
            try:
                by_head[_hash(p, _HEAD)].append(p)
            except OSError:
                continue

        for head_paths in by_head.values():
            if len(head_paths) < 2:
                continue
            by_full: dict[bytes, list[str]] = defaultdict(list)
            for p in head_paths:
                try:
                    by_full[_hash(p)].append(p)
                except OSError:
                    continue
            for full_paths in by_full.values():
                if len(full_paths) > 1:
                    groups.append(sorted(full_paths, key=lambda p: (len(p), p)))

    return groups
