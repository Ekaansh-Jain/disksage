"""Safety-focused tests. These prove the guarantees that matter when someone
else runs the tool on their own machine:

  - the knowledge base classifies known things correctly,
  - the safety floor blocks protected/out-of-home paths,
  - duplicate detection is EXACT (never groups non-identical files) and never
    proposes deleting an only copy,
  - non-overlapping sizing doesn't double-count.

Run:  ./.venv/bin/python -m pytest -q
"""

import os
import tempfile

from disksage import dedup, knowledge, llm, safety
from disksage.scan import dir_size

HOME = os.path.expanduser("~")


# ---- knowledge base ----

def test_known_caches_are_safe():
    assert knowledge.look_up(os.path.join(HOME, "Library/Caches/pip")).tier == "safe"

def test_node_modules_is_review():
    assert knowledge.look_up("/anywhere/project/node_modules").tier == "review"

def test_keychains_are_keep():
    assert knowledge.look_up(os.path.join(HOME, "Library/Keychains")).tier == "keep"

def test_file_kind_labels_models_and_defaults():
    assert knowledge.file_kind("/x/model.safetensors").label == "Large ML model"
    assert knowledge.file_kind("/x/model.safetensors").tier == "review"
    assert knowledge.file_kind("/x/weird.qqq").label == "Large file"


# ---- safety floor ----

def test_home_and_top_level_are_blocked():
    for p in (HOME, os.path.join(HOME, "Documents"),
              os.path.join(HOME, "Library/Keychains")):
        ok, _ = safety.is_safe_to_trash(p)
        assert ok is False, p

def test_outside_home_is_blocked():
    ok, _ = safety.is_safe_to_trash("/etc/hosts")
    assert ok is False

def test_ordinary_folder_in_home_is_allowed():
    d = tempfile.mkdtemp(dir=HOME, prefix=".reclaim_test_")
    try:
        ok, reason = safety.is_safe_to_trash(d)
        assert ok is True, reason
    finally:
        os.rmdir(d)


# ---- duplicate detection ----

def _write(path, data):
    with open(path, "wb") as f:
        f.write(data)

def test_dedup_groups_only_identical(tmp_path):
    a = tmp_path / "a.bin"; _write(a, b"X" * 100_000)
    b = tmp_path / "b.bin"; _write(b, b"X" * 100_000)      # identical to a
    c = tmp_path / "c.bin"; _write(c, b"Y" * 100_000)      # same size, differs
    d = tmp_path / "d.bin"; _write(d, b"Z" * 50_000)       # unique size

    files = [(str(p), p.stat().st_size) for p in (a, b, c, d)]
    groups = dedup.find_duplicates(files)

    assert len(groups) == 1
    assert set(groups[0]) == {str(a), str(b)}          # only true copies grouped
    assert str(c) not in sum(groups, [])               # different content excluded
    assert str(d) not in sum(groups, [])               # unique size excluded

def test_dedup_keeps_an_original(tmp_path):
    a = tmp_path / "a.bin"; _write(a, b"D" * 200_000)
    b = tmp_path / "b.bin"; _write(b, b"D" * 200_000)
    groups = dedup.find_duplicates([(str(a), a.stat().st_size),
                                    (str(b), b.stat().st_size)])
    # group[0] is the kept original; only the rest are ever deletable.
    assert len(groups[0]) == 2
    assert len(groups[0][1:]) == 1


# ---- local LLM provider resolution ----

def test_local_provider_resolves_from_env(monkeypatch):
    # An explicitly configured local server resolves without any network call
    # (model provided), and takes priority when forced.
    monkeypatch.setenv("DISKSAGE_PROVIDER", "local")
    monkeypatch.setenv("DISKSAGE_LOCAL_BASE_URL", "http://localhost:11434/v1")
    monkeypatch.setenv("DISKSAGE_LOCAL_MODEL", "llama3.1")
    cfg = llm.resolve()
    assert cfg is not None
    assert cfg["name"] == "local"
    assert cfg["base_url"].endswith("11434/v1")
    assert cfg["model"] == "llama3.1"
    assert cfg["api_key"] == "local"


# ---- non-overlapping sizing ----

def test_dir_size_excludes_stopped_children(tmp_path):
    child = tmp_path / "child"
    child.mkdir()
    _write(tmp_path / "top.bin", b"A" * 1000)
    _write(child / "inner.bin", b"B" * 5000)

    full = dir_size(str(tmp_path))
    own = dir_size(str(tmp_path), stop={str(child)})
    assert full == 6000
    assert own == 1000
