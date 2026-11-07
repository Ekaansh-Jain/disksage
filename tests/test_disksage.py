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

from disksage import cli, dedup, knowledge, llm, safety
from disksage.classify import build_file_items
from disksage.scan import collect_files, dir_size, is_dataless, scan_dirs, size_on_disk

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
    a = tmp_path / "a.bin"
    _write(a, b"X" * 100_000)
    b = tmp_path / "b.bin"
    _write(b, b"X" * 100_000)      # identical to a
    c = tmp_path / "c.bin"
    _write(c, b"Y" * 100_000)      # same size, differs
    d = tmp_path / "d.bin"
    _write(d, b"Z" * 50_000)       # unique size

    files = [(str(p), p.stat().st_size) for p in (a, b, c, d)]
    groups = dedup.find_duplicates(files)

    assert len(groups) == 1
    assert set(groups[0]) == {str(a), str(b)}          # only true copies grouped
    assert str(c) not in sum(groups, [])               # different content excluded
    assert str(d) not in sum(groups, [])               # unique size excluded

def test_dedup_keeps_an_original(tmp_path):
    a = tmp_path / "a.bin"
    _write(a, b"D" * 200_000)
    b = tmp_path / "b.bin"
    _write(b, b"D" * 200_000)
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

    top = size_on_disk(os.stat(tmp_path / "top.bin"))
    inner = size_on_disk(os.stat(child / "inner.bin"))
    full = dir_size(str(tmp_path))
    own = dir_size(str(tmp_path), stop={str(child)})
    assert full == top + inner
    assert own == top


# ---- sizes are bytes on disk, each file counted once ----

MB = 1024 * 1024


def test_sparse_file_counts_bytes_on_disk(tmp_path):
    sparse = tmp_path / "vm.img"
    with open(sparse, "wb") as f:
        f.truncate(512 * MB)          # 512 MB long, almost nothing on disk
    assert os.stat(sparse).st_size == 512 * MB
    assert dir_size(str(tmp_path)) < MB


def test_hard_link_counted_once_across_a_scan(tmp_path):
    (tmp_path / "one").mkdir()
    (tmp_path / "two").mkdir()
    _write(tmp_path / "one" / "model.bin", os.urandom(2 * MB))
    os.link(tmp_path / "one" / "model.bin", tmp_path / "two" / "model.bin")
    seen: set = set()
    assert dir_size(str(tmp_path / "one"), seen=seen) >= 2 * MB
    assert dir_size(str(tmp_path / "two"), seen=seen) == 0


def test_hard_link_is_never_its_own_duplicate(tmp_path):
    _write(tmp_path / "weights.safetensors", os.urandom(2 * MB))
    os.link(tmp_path / "weights.safetensors", tmp_path / "weights-link.safetensors")
    os.symlink(tmp_path / "weights.safetensors", tmp_path / "weights-symlink.safetensors")
    files = collect_files(MB, roots=[str(tmp_path)])
    assert len(files) == 1                     # one real file, whatever its names
    items = build_file_items(files, big_min=100 * MB)
    assert not [i for i in items if i.tier == "duplicate"]


def test_real_duplicates_are_found_but_not_pre_ticked(tmp_path):
    data = os.urandom(2 * MB)
    (tmp_path / "a").mkdir()
    (tmp_path / "b").mkdir()
    _write(tmp_path / "a" / "data.parquet", data)
    _write(tmp_path / "b" / "data.parquet", data)
    items = build_file_items(collect_files(MB, roots=[str(tmp_path)]), big_min=100 * MB)
    dups = [i for i in items if i.tier == "duplicate"]
    assert len(dups) == 1
    assert dups[0].size >= 2 * MB
    assert cli._default_checked(dups[0]) is False


def test_cloud_placeholders_are_recognized():
    class FakeStat:
        st_flags = 0x40000000
    assert is_dataless(FakeStat())
    FakeStat.st_flags = 0
    assert not is_dataless(FakeStat())


def test_scan_dirs_reports_bytes_on_disk(tmp_path):
    proj = tmp_path / "proj" / "node_modules"
    proj.mkdir(parents=True)
    with open(proj / "sparse.bin", "wb") as f:
        f.truncate(300 * MB)
    _write(proj / "real.bin", os.urandom(3 * MB))
    items = scan_dirs(min_size=MB, root=str(tmp_path))
    assert [os.path.basename(i.path) for i in items] == ["node_modules"]
    assert 3 * MB <= items[0].size < 10 * MB


# ---- safety floor: packages, synced folders, credentials ----

def _make(path, data=b"x"):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    _write(path, data)
    return path


def test_nothing_inside_a_photos_library_can_go():
    lib = os.path.join(HOME, "Documents", "Photos Library.photoslibrary")
    photo = _make(os.path.join(lib, "originals", "A", "IMG_0001.mov"))
    assert safety.is_safe_to_trash(lib)[0] is False
    assert safety.is_safe_to_trash(photo)[0] is False
    assert safety.is_safe_to_trash(os.path.dirname(photo))[0] is False


def test_package_contents_are_never_scanned():
    _make(os.path.join(HOME, "Documents", "Trip.photoslibrary", "originals", "big.mov"),
          os.urandom(2 * MB))
    _make(os.path.join(HOME, "Documents", "Tool.app", "Contents", "node_modules", "x.js"),
          os.urandom(2 * MB))
    files = collect_files(MB, roots=[os.path.join(HOME, "Documents")])
    assert not [f for f in files if ".photoslibrary" in f.path or ".app" in f.path]
    folders = scan_dirs(min_size=MB, root=os.path.join(HOME, "Documents"))
    assert not [i for i in folders if ".app" in i.path]


def test_synced_folders_backups_and_credentials_are_protected():
    for rel in ("Library/Mobile Documents/com~apple~CloudDocs/report.pdf",
                "Library/CloudStorage/Dropbox/notes.txt",
                "Library/Application Support/MobileSync/Backup/00008030/Manifest.db",
                ".ssh/id_ed25519",
                ".aws/credentials",
                ".config/gh/hosts.yml",
                ".cache/huggingface/token"):
        path = _make(os.path.join(HOME, rel))
        assert safety.is_safe_to_trash(path)[0] is False, rel


def test_model_caches_are_review_not_pre_ticked():
    for rel in (".cache/huggingface/hub", ".cache/huggingface/datasets", ".cache/torch"):
        hit = knowledge.look_up(os.path.join(HOME, rel))
        assert hit.tier == "review", rel
    assert knowledge.look_up(os.path.join(HOME, ".cache/huggingface/xet")).tier == "safe"
    assert knowledge.look_up(
        os.path.join(HOME, "Library/Application Support/MobileSync")).tier == "keep"


def test_huggingface_folder_is_offered_piece_by_piece():
    from disksage.scan import collect_candidates
    for rel in ("hub/models--x--y/blobs/a", "datasets/glue/data.arrow", "token"):
        _make(os.path.join(HOME, ".cache/huggingface", rel))
    candidates = collect_candidates()
    assert os.path.join(HOME, ".cache/huggingface") not in candidates
    assert os.path.join(HOME, ".cache/huggingface/hub") in candidates
    assert os.path.join(HOME, ".cache/huggingface/datasets") in candidates


def test_turbo_label():
    assert knowledge.look_up("/x/proj/.turbo").label == "Turbo cache"


# ---- git-tracked folders are source, not build output ----

def _git(*args, cwd):
    import subprocess
    subprocess.run(["git", *args], cwd=cwd, check=True, capture_output=True)


def test_git_tracked_build_folder_is_flagged_and_never_pre_ticked():
    repo = os.path.join(HOME, "Developer", "app")
    _make(os.path.join(repo, "build", "release.sh"), b"echo release\n")
    _make(os.path.join(repo, "dist", "bundle.js"), b"// generated\n")
    _git("init", "-q", cwd=repo)
    _git("add", "build", cwd=repo)
    _git("-c", "user.email=t@example.com", "-c", "user.name=t", "commit", "-qm", "init", cwd=repo)

    assert safety.tracked_file_count(os.path.join(repo, "build")) == 1
    assert safety.tracked_file_count(os.path.join(repo, "dist")) == 0   # untracked output
    assert safety.tracked_file_count(os.path.join(HOME, "Documents")) == 0  # not a repo

    from disksage.scan import Item
    items = [Item(path=os.path.join(repo, "build"), size=1, tier="review", source="knowledge"),
             Item(path=os.path.join(repo, "dist"), size=1, tier="review", source="knowledge")]
    cli._flag_project_folders(items)
    assert items[0].tracked == 1 and "tracked by git" in items[0].why
    assert items[1].tracked == 0
    items[0].tier = "safe"
    assert cli._default_checked(items[0]) is False


# ---- clean frees space for real ----

def test_remove_regenerable_deletes_a_cache_but_never_a_symlink_target(tmp_path):
    cache = tmp_path / "cache"
    cache.mkdir()
    _write(cache / "a.bin", b"x" * 100)
    safety.remove_regenerable(str(cache))
    assert not cache.exists()

    data = tmp_path / "real-data"
    data.mkdir()
    _write(data / "keep.txt", b"important")
    link = tmp_path / "link-to-data"
    os.symlink(data, link)
    safety.remove_regenerable(str(link))
    assert not os.path.lexists(link)
    assert (data / "keep.txt").exists()


def test_trash_folder_is_never_offered():
    from disksage.scan import SINGLETONS
    assert os.path.join(HOME, ".Trash") not in SINGLETONS
    assert knowledge.look_up(os.path.join(HOME, ".Trash")) is None


def test_only_built_in_safe_caches_are_deleted_directly():
    from disksage.scan import Item
    cache = Item(path="/x/pip", size=1, tier="safe", source="knowledge")
    ai_cache = Item(path="/x/thing", size=1, tier="safe", source="ai")
    venv = Item(path="/x/.venv", size=1, tier="review", source="knowledge")
    dup = Item(path="/x/copy.bin", size=1, tier="duplicate", source="hash")
    assert cli._deletes_directly(cache)
    assert not any(cli._deletes_directly(i) for i in (ai_cache, venv, dup))


def test_docker_and_tool_tips_need_the_tool(monkeypatch):
    from disksage import tools
    monkeypatch.setattr(tools.shutil, "which", lambda _: None)
    assert tools.docker_usage() is None
    assert tools.available() == []
    for cleanup in tools.NATIVE_CLEANUPS:
        assert cleanup.command[0] == cleanup.tool and cleanup.removes


def _answer(prompt=""):
    # Non-interactive `clean`: delete known caches, trash node_modules, say no to the rest.
    if "known caches" in prompt:
        return "y"
    if "node_modules" in prompt:
        return "y"
    if "Type 'yes'" in prompt:
        return "yes"
    return "n"


def _clutter():
    pip_cache = os.path.join(HOME, "Library/Caches/pip")
    _make(os.path.join(pip_cache, "http", "wheel.bin"), os.urandom(3 * MB))
    node_modules = os.path.join(HOME, "Projects", "shop", "node_modules")
    _make(os.path.join(node_modules, "big.bin"), os.urandom(3 * MB))
    return pip_cache, node_modules


def test_clean_deletes_caches_and_trashes_the_rest(monkeypatch, capsys):
    from disksage import tools
    pip_cache, node_modules = _clutter()
    trashed = []
    monkeypatch.setattr(cli, "trash", trashed.append)
    monkeypatch.setattr(tools, "docker_usage", lambda: None)
    monkeypatch.setattr("builtins.input", _answer)

    assert cli.main(["clean", "--no-ai", "--no-files", "--min-size", "1"]) == 0

    out = capsys.readouterr().out
    assert not os.path.exists(pip_cache)          # deleted directly: it rebuilds itself
    assert trashed == [node_modules]              # yours to get back: went to the Trash
    assert "Free space went up by" in out


def test_clean_dry_run_changes_nothing(monkeypatch, capsys):
    from disksage import tools
    pip_cache, node_modules = _clutter()
    trashed = []
    monkeypatch.setattr(cli, "trash", trashed.append)
    monkeypatch.setattr(tools, "docker_usage", lambda: None)
    monkeypatch.setattr("builtins.input", _answer)

    assert cli.main(["clean", "--dry-run", "--no-ai", "--no-files", "--min-size", "1"]) == 0

    assert os.path.exists(pip_cache) and os.path.exists(node_modules)
    assert trashed == []
    assert "Dry run" in capsys.readouterr().out
