"""Command line interface.

    disksage            in a terminal: scan, show what can be freed, then offer
                        to clean right away using the same scan. When piped,
                        it's the read-only report.
    disksage scan       read-only report: a summary first, `--details` for every
                        item, `--json` for scripts and agents
    disksage clean      pick what to remove from a checklist: caches that rebuild
                        themselves are deleted, everything else goes to the
                        Trash. Reuses a scan from the last 10 minutes
                        (`--rescan` to scan again)
    disksage models     every local model, and the ones stored by more than one app
    disksage dedupe     store duplicate models once, as APFS clones
    disksage setup      pick an AI provider for unrecognized folders (optional)
    disksage doctor     show the AI provider and the local model servers it sees

Flags (scan, clean):
    --min-size MB      ignore folders smaller than this (default 20)
    --provider NAME    force local | groq | gemini
    --no-ai            built-in list only, never call an AI
    --root PATH        scan only this folder, for project folders
                       (node_modules, .venv, build…) and big files
    --dry-run          (clean) show what would happen, change nothing
"""

from __future__ import annotations

import argparse
import contextlib
import io
import json
import os
import shutil
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from contextlib import contextmanager
from dataclasses import asdict, dataclass, field, fields

from . import __version__, config, knowledge, llm, models, tools
from .classify import annotate_known, annotate_with_ai, build_file_items, human
from .safety import (
    empty_trash,
    git_risk,
    is_safe_to_trash,
    remove_regenerable,
    tracked_file_count,
    trash,
)
from .scan import (
    CRUFT_DIR_NAMES,
    PROJECT_ROOTS,
    Item,
    ScanProgress,
    project_last_touched,
    scan_dirs,
    walk_projects,
)

HOME = os.path.expanduser("~")
_REAL_HOME = os.path.realpath(HOME)
MB = 1024 * 1024

TIER_STYLE = {
    "safe":      ("green",     "SAFE  "),
    "duplicate": ("magenta",   "DUP   "),
    "review":    ("yellow",    "REVIEW"),
    "keep":      ("red",       "KEEP  "),
}
TIER_ORDER = {"safe": 0, "duplicate": 1, "review": 2, "keep": 3}

# `clean` reuses a scan this recent, so `scan` followed by `clean` doesn't make
# you wait twice.
SCAN_REUSE_SECONDS = 10 * 60

# Stand-in path for the checklist entry that runs Docker's own cleanup.
DOCKER_ITEM = "docker:system-prune"


def _console():
    try:
        from rich.console import Console
        return Console()
    except ImportError:
        return None


def _free_space() -> str:
    usage = shutil.disk_usage("/")
    return f"{human(usage.free)} free of {human(usage.total)}"


def _free_bytes() -> int:
    return shutil.disk_usage(HOME).free


def _short_path(path: str) -> str:
    """~/x instead of /Users/you/x (also when HOME sits behind a symlink)."""
    for home in (HOME, _REAL_HOME):
        if path == home:
            return "~"
        if path.startswith(home + os.sep):
            return "~" + path[len(home):]
    return path


def _display_path(it) -> str:
    return tools.DOCKER_PRUNE.display if it.kind == "command" else _short_path(it.path)


def _whole_tail(parts: list[str], room: int) -> list[str]:
    """As many whole names from the end of `parts` as fit in `room` columns."""
    tail: list[str] = []
    for part in reversed(parts):
        if sum(map(len, tail)) + len(tail) + len(part) > room:
            break
        tail.insert(0, part)
    return tail


def _fit(text: str, budget: int) -> str:
    """Shorten to `budget` columns by dropping the middle, never cutting a
    folder name in half. The deepest folders say the most (which project,
    which cache), so they win over the start of the path:
    …/old-react-app/node_modules, or ~/Library/…/Service Worker when the start
    still fits."""
    if len(text) <= budget:
        return text
    parts = text.split(os.sep)
    if len(parts) < 3:
        return "…" + text[-(budget - 1):]
    head = os.sep.join(parts[:2])
    with_head = _whole_tail(parts[2:], budget - len(head) - 3)   # head + "/…/" + tail
    without = _whole_tail(parts[1:], budget - 2)                 # "…/" + tail
    if with_head and len(with_head) >= len(without):
        return f"{head}{os.sep}…{os.sep}{os.sep.join(with_head)}"
    if without:
        return f"…{os.sep}{os.sep.join(without)}"
    return "…" + text[-(budget - 1):]


def _columns() -> int:
    return shutil.get_terminal_size((100, 24)).columns


def _say(con, rich_msg: str, plain_msg: str) -> None:
    # soft_wrap: let the terminal wrap long lines. Rich would otherwise insert
    # hard line breaks, splitting paths and commands people copy and paste.
    con.print(rich_msg, soft_wrap=True) if con else print(plain_msg)


@contextmanager
def _status(con, text: str):
    """Show a spinner (rich) or a plain line while a slow step runs."""
    if con:
        with con.status(text):
            yield
    else:
        print(text)
        yield


@contextmanager
def _live_progress(con, progress: ScanProgress, label: str):
    """A spinner with a running count of files and bytes, refreshed a few
    times a second, so a long scan never looks stuck."""
    if not con:
        print(f"{label}…")
        yield
        return
    with con.status(f"{label}…") as status:
        done = threading.Event()

        def refresh():
            while not done.wait(0.2):
                status.update(f"{label}… {progress.files:,} files · "
                              f"{human(progress.bytes)} counted")

        ticker = threading.Thread(target=refresh, daemon=True)
        ticker.start()
        try:
            yield
        finally:
            done.set()
            ticker.join()


@contextmanager
def _bytes_progress(con, total: int, label: str):
    """A progress bar for reading big files (hashing models). Yields a
    callback that takes the number of bytes just read."""
    if not con or total <= 0 or not con.is_terminal:
        yield lambda n: None
        return
    from rich.progress import (
        BarColumn,
        DownloadColumn,
        Progress,
        TextColumn,
        TimeRemainingColumn,
        TransferSpeedColumn,
    )
    columns = (TextColumn("{task.description}"), BarColumn(), DownloadColumn(),
               TransferSpeedColumn(), TimeRemainingColumn())
    with Progress(*columns, console=con, transient=True) as bar:
        task = bar.add_task(label, total=total)
        yield lambda n: bar.advance(task, n)


def _ago_short(ts: float) -> str:
    seconds = max(0, int(time.time() - ts))
    if seconds < 60:
        return "just now"
    minutes = seconds // 60
    return f"{minutes} minute{'s' if minutes != 1 else ''} ago"


def _seconds(s: float) -> str:
    return f"{s:.1f} s" if s < 10 else f"{s:.0f} s"


# ---- AI setup ----

def _custom_local(questionary) -> dict:
    base = questionary.text(
        "Base URL of the OpenAI-compatible server "
        "(e.g. http://localhost:11434/v1):").ask()
    if not base or not base.strip():
        return {}
    base = base.strip()
    if not base.startswith(("http://", "https://")):
        base = "http://" + base
    found = llm.list_models(base)
    if found:
        model = found[0] if len(found) == 1 else (
            questionary.select("Which model?", choices=found).ask() or found[0])
    else:
        print("(couldn't list models from that URL — type the model name to use)")
        model = questionary.text("Model name:").ask()
        if not model or not model.strip():
            return {}
        model = model.strip()
    return {"DISKSAGE_PROVIDER": "local",
            "DISKSAGE_LOCAL_BASE_URL": base, "DISKSAGE_LOCAL_MODEL": model}


def _setup_local(questionary) -> dict:
    print("Scanning for local LLM servers…")
    servers = llm.detect_local_servers()
    if servers:
        labels = [f"{s['name']}  —  {s['base_url']}  ({len(s['models'])} model(s))"
                  for s in servers]
        labels.append("Enter a custom URL…")
        pick = questionary.select("Found these local servers:", choices=labels).ask()
        if pick is None:
            return {}
        if pick.startswith("Enter a custom"):
            return _custom_local(questionary)
        s = servers[labels.index(pick)]
        model = s["models"][0]
        if len(s["models"]) > 1:
            model = questionary.select("Which model?", choices=s["models"]).ask() or model
        print(f"→ Using {s['name']} at {s['base_url']} with {model}")
        return {"DISKSAGE_PROVIDER": "local",
                "DISKSAGE_LOCAL_BASE_URL": s["base_url"], "DISKSAGE_LOCAL_MODEL": model}

    print("No local server found on the usual ports.")
    print("Start one first (e.g. `ollama serve`, LM Studio, or `llama-server`).")
    if questionary.confirm("Enter a custom server URL instead?", default=False).ask():
        return _custom_local(questionary)
    return {}


def cmd_doctor(args=None) -> int:
    """Show the active provider and which local servers are reachable."""
    print(f"Active AI provider: {llm.describe()}")
    print("Scanning local ports (Ollama, LM Studio, llama.cpp, Jan, …)…")
    servers = llm.detect_local_servers()
    if not servers:
        print("  no local OpenAI-compatible servers found on the usual ports.")
        return 0
    for s in servers:
        shown = ", ".join(s["models"][:3]) + (" …" if len(s["models"]) > 3 else "")
        print(f"  ✓ {s['name']}: {s['base_url']}  →  {shown}")
    return 0


def cmd_setup(args=None, first_run: bool = False) -> int:
    """Interactive first-run / re-run setup: pick an AI provider (or none) and,
    if needed, paste a key. Saved to the user config, never printed back."""
    if not sys.stdin.isatty():
        print("Run `disksage setup` in an interactive terminal.")
        return 1
    try:
        import questionary
    except ImportError:
        print("Setup needs the `questionary` package (pip install questionary).")
        return 1

    print("Set up how disksage identifies unknown files.\n"
          "It only ever sends file paths + sizes to the provider — never contents.\n"
          "(You can re-run this anytime with `disksage setup`.)\n")

    choice = questionary.select(
        "Choose an AI provider:",
        choices=[
            "Groq — free cloud API (paste a key)",
            "Google Gemini — free cloud API (paste a key)",
            "Local — auto-detect (Ollama, LM Studio, llama.cpp, …)",
            "No AI — use the built-in knowledge base only",
        ],
    ).ask()
    if choice is None:
        print("Setup cancelled. Nothing saved.")
        return 1

    values: dict[str, str] = {}
    if choice.startswith("Groq"):
        key = questionary.password(
            "Paste your Groq API key (free at console.groq.com):").ask()
        if key and key.strip():
            values = {"DISKSAGE_PROVIDER": "groq", "GROQ_API_KEY": key.strip(),
                      "DISKSAGE_GROQ_MODEL": "openai/gpt-oss-20b"}
    elif choice.startswith("Google"):
        key = questionary.password(
            "Paste your Gemini API key (free at aistudio.google.com):").ask()
        if key and key.strip():
            values = {"DISKSAGE_PROVIDER": "gemini", "GEMINI_API_KEY": key.strip(),
                      "DISKSAGE_GEMINI_MODEL": "gemini-flash-latest"}
    elif choice.startswith("Local"):
        values = _setup_local(questionary)

    if not values:
        # "No AI", or a prompt left blank — record the choice so we don't ask again.
        values = {"DISKSAGE_AI": "off"}

    path = config.save(values)
    _invalidate_scan()  # labels may change
    print(f"\nSaved to {path}")
    print(f"AI provider: {llm.describe()}")
    if not first_run:
        print("\nNext:  disksage   (scan, then choose what to clean)")
    return 0


# ---- project folders: git-tracked or stale ----

# Project folders that rebuild from the project's own files (npm install, a
# fresh venv, the next build). Only these get pre-ticked, and only in projects
# nobody has touched for STALE_DAYS; they still go to the Trash, not deleted.
REBUILDABLE = {"node_modules", ".venv", "venv", "target", "build", "dist",
               ".next", ".nuxt", ".gradle"}
STALE_DAYS = 30


def _ago(ts: float) -> str:
    days = int((time.time() - ts) // 86400)
    if days < 1:
        return "today"
    if days < 60:
        return f"{days} day{'s' if days != 1 else ''} ago"
    if days < 730:
        return f"{days // 30} months ago"
    return f"{days // 365} years ago"


def _is_stale(it) -> bool:
    touched = getattr(it, "touched", 0.0)
    return bool(touched) and time.time() - touched > STALE_DAYS * 86400


def _flag_project_folders(items) -> None:
    """Annotate project folders (node_modules, build/, .venv, …):

    - A folder named like build output can still hold source code that git
      tracks. Flag those so they're never pre-ticked and the user sees why.
    - Otherwise note when the project was last worked on, so only stale
      projects get their rebuildable folders pre-ticked.

    The git checks are one `git ls-files` process per folder. They wait on
    git, not on Python, so they run side by side."""
    folders = [it for it in items
               if it.kind == "dir" and os.path.basename(it.path) in CRUFT_DIR_NAMES]
    if len(folders) > 1:
        with ThreadPoolExecutor(max_workers=8) as pool:
            counts = list(pool.map(lambda it: tracked_file_count(it.path), folders))
    else:
        counts = [tracked_file_count(it.path) for it in folders]
    for it, count in zip(folders, counts, strict=True):
        name = os.path.basename(it.path)
        it.tracked = count
        if it.tracked:
            it.tier = "review"
            it.why = (f"⚠ {it.tracked} file(s) here are tracked by git: "
                      f"this is source, not build output")
            continue
        it.touched = project_last_touched(it.path)
        if it.touched and name in REBUILDABLE:
            it.why = f"{it.why}; project last touched {_ago(it.touched)}"


def _default_checked(it) -> bool:
    # Pre-tick only the low-risk items: knowledge-base safe caches. Duplicates
    # start unchecked: two identical files in different projects may each be
    # used by code, so the user decides. Review and AI-tagged items start
    # unchecked so the user opts in, and so does anything git tracks.
    if getattr(it, "tracked", 0):
        return False
    if it.tier == "safe":
        return it.source != "ai"
    # Rebuildable project folders, in projects nobody has touched for a month.
    return (it.tier == "review" and it.source == "knowledge" and it.kind == "dir"
            and os.path.basename(it.path) in REBUILDABLE and _is_stale(it))


def _deletes_directly(it) -> bool:
    """Caches that rebuild themselves (SAFE in the built-in list) are deleted
    outright: in the Trash they'd free nothing until it's emptied. Anything
    that could be yours (REVIEW, duplicates, AI-labelled items) goes to the
    Trash instead."""
    return it.tier == "safe" and it.source == "knowledge" and not getattr(it, "tracked", 0)


# ---- scanning ----

@dataclass
class ScanResult:
    items: list
    files_seen: int = 0
    seconds: float = 0.0
    created: float = field(default_factory=time.time)
    cloud_only_files: int = 0
    cloud_only_bytes: int = 0
    unknown_count: int = 0        # folders neither the built-in list nor the AI identified
    unknown_bytes: int = 0
    suggest_setup: bool = False   # no AI configured, and it could have helped
    docker: tools.DockerReport | None = None
    dupes: models.DuplicateEstimate | None = None
    options: dict = field(default_factory=dict)


def _scan_options(args) -> dict:
    """What a reused scan must have been run with."""
    return {
        "min_size": args.min_size, "big_size": args.big_size, "dup_size": args.dup_size,
        "no_files": bool(args.no_files), "no_ai": bool(args.no_ai),
        "root": os.path.realpath(args.root) if args.root else None,
        "provider": os.environ.get("DISKSAGE_PROVIDER", ""),
    }


def _estimate_model_duplicates() -> models.DuplicateEstimate | None:
    """Fast (no file is read), and optional: a failure here must never cost the
    user their scan."""
    try:
        return models.estimate_duplicates(models.all_model_files())
    except Exception:
        return None


def _print_provider_banner(con) -> None:
    cfg = llm.resolve()
    if cfg:
        msg = (f"AI: {cfg['name']} ({cfg['model']}) labels folders the built-in list "
               f"doesn't know. It sees only paths and sizes, never file contents.")
        con.print(f"[dim]{msg}[/dim]") if con else print(msg)


def _scan(args, con) -> ScanResult:
    """Scan, classify and remember the result for `clean`."""
    _print_provider_banner(con)
    progress = ScanProgress()
    roots = [args.root] if args.root else PROJECT_ROOTS
    file_min = None if args.no_files else args.dup_size * MB

    with ThreadPoolExecutor(max_workers=2) as side:
        # Docker and the model stores are asked in parallel with the file scan.
        # (A --root scan is about one folder, so skip them there.)
        docker_job = None if args.root else side.submit(tools.docker_report)
        dupes_job = None if args.root else side.submit(_estimate_model_duplicates)
        with _live_progress(con, progress, "Scanning"):
            walk = walk_projects(roots, file_min=file_min, progress=progress)
            dir_items = scan_dirs(min_size=args.min_size * MB, root=args.root,
                                  cruft=walk.cruft, progress=progress)
        docker = docker_job.result() if docker_job else None
        dupes = dupes_job.result() if dupes_job else None

    unknown = annotate_known(dir_items)
    _flag_project_folders(dir_items)
    provider = None if args.no_ai else llm.resolve()
    if unknown and provider:
        with _status(con, f"Identifying {len(unknown)} unrecognized folder(s) "
                          f"with {provider['name']}…"):
            outcome = annotate_with_ai(unknown)
        if outcome.ok:
            _say(con, f"[green]✓ Identified {outcome.labeled}/{outcome.total} unknown "
                      f"folder(s) with {provider['name']}.[/green]",
                 f"Identified {outcome.labeled}/{outcome.total} unknown folders "
                 f"with {provider['name']}.")
        else:
            _say(con, f"[yellow]⚠ AI unavailable: {outcome.message}. "
                      f"{outcome.total} folder(s) stay REVIEW.[/yellow]",
                 f"AI unavailable: {outcome.message}. {outcome.total} folders stay REVIEW.")

    file_items = (build_file_items(walk.files, big_min=args.big_size * MB)
                  if file_min is not None else [])
    items = dir_items + file_items
    still_unknown = [i for i in items if i.source == "unknown"]
    result = ScanResult(
        items=items, files_seen=progress.files, seconds=progress.seconds,
        cloud_only_files=walk.cloud_only_files, cloud_only_bytes=walk.cloud_only_bytes,
        unknown_count=len(still_unknown), unknown_bytes=sum(i.size for i in still_unknown),
        suggest_setup=bool(still_unknown) and not args.no_ai and provider is None,
        docker=docker, dupes=dupes, options=_scan_options(args))
    _say(con, f"[dim]Scanned {progress.files:,} files in {_seconds(progress.seconds)}.[/dim]",
         f"Scanned {progress.files:,} files in {_seconds(progress.seconds)}.")
    _save_scan(result)
    return result


# ---- remembering the last scan ----

def _scan_cache_path() -> str:
    return os.path.join(models.STATE_DIR, "last-scan.json")


def _save_scan(result: ScanResult) -> None:
    data = {
        "version": __version__,
        "created": result.created,
        "options": result.options,
        "files_seen": result.files_seen,
        "seconds": result.seconds,
        "cloud_only_files": result.cloud_only_files,
        "cloud_only_bytes": result.cloud_only_bytes,
        "unknown_count": result.unknown_count,
        "unknown_bytes": result.unknown_bytes,
        "suggest_setup": result.suggest_setup,
        "docker": ({"rows": [list(r) for r in result.docker.rows],
                    "reclaimable": result.docker.reclaimable} if result.docker else None),
        "dupes": asdict(result.dupes) if result.dupes else None,
        "items": [asdict(i) for i in result.items],
    }
    path = _scan_cache_path()
    try:
        os.makedirs(os.path.dirname(path), exist_ok=True)
        tmp = f"{path}.{os.getpid()}.tmp"
        # It lists paths on your disk: readable by you only.
        with open(os.open(tmp, os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o600), "w") as f:
            json.dump(data, f)
        os.replace(tmp, path)
    except OSError:
        pass  # remembering is a convenience; never fail a scan over it


def _invalidate_scan() -> None:
    with contextlib.suppress(OSError):
        os.remove(_scan_cache_path())


_ITEM_FIELDS = {f.name for f in fields(Item)}


def _rederive(items: list) -> None:
    """Tiers come from the rules, never from the saved file: only paths and
    sizes are reused. An old or edited file can't make anything SAFE."""
    for it in items:
        if it.kind == "dir" and it.source == "knowledge":
            hit = knowledge.look_up(it.path)
            if hit:
                it.tier, it.label, it.why = hit.tier, hit.label, hit.why
            else:
                it.tier, it.label, it.why, it.source = (
                    "review", "unrecognized", "not in knowledge base", "unknown")
        elif it.kind == "duplicate":
            it.tier = "duplicate"
        elif it.tier != "keep":
            it.tier = "review"
        it.tracked, it.touched = 0, 0.0
    _flag_project_folders(items)


def _load_scan(args) -> ScanResult | None:
    """The last scan, if it's recent, was made with the same options, and by
    this version. Items whose files are gone are dropped."""
    try:
        with open(_scan_cache_path()) as f:
            data = json.load(f)
        if data.get("version") != __version__ or data.get("options") != _scan_options(args):
            return None
        created = float(data.get("created", 0))
        if not 0 <= time.time() - created <= SCAN_REUSE_SECONDS:
            return None
        items = []
        for raw in data.get("items", []):
            if not isinstance(raw, dict):
                continue
            it = Item(**{k: v for k, v in raw.items() if k in _ITEM_FIELDS})
            # Only plain absolute paths that still exist; scans never produce "..".
            if (isinstance(it.path, str) and os.path.isabs(it.path)
                    and os.path.normpath(it.path) == it.path and os.path.lexists(it.path)
                    and isinstance(it.size, int)):
                items.append(it)
        _rederive(items)
        docker = data.get("docker")
        dupes = data.get("dupes")
        return ScanResult(
            items=items, files_seen=int(data.get("files_seen", 0)),
            seconds=float(data.get("seconds", 0)), created=created,
            cloud_only_files=int(data.get("cloud_only_files", 0)),
            cloud_only_bytes=int(data.get("cloud_only_bytes", 0)),
            unknown_count=int(data.get("unknown_count", 0)),
            unknown_bytes=int(data.get("unknown_bytes", 0)),
            suggest_setup=bool(data.get("suggest_setup")),
            docker=(tools.DockerReport(tuple(tuple(r) for r in docker["rows"]),
                                       int(docker["reclaimable"])) if docker else None),
            dupes=models.DuplicateEstimate(**dupes) if dupes else None,
            options=data["options"])
    except (OSError, ValueError, TypeError, KeyError):
        return None


# ---- the summary: what can be freed, grouped by what it costs you ----

@dataclass
class Bucket:
    key: str
    size: int
    effect: str
    what: str
    approx: bool = False   # an estimate (duplicate models matched by size, not yet hashed)


BUCKET_STYLE = {"now": "green", "dedupe": "green", "stale": "yellow",
                "models": "yellow", "docker": "cyan", "judge": "white"}

# Checklist sections, in the order the summary lists them.
SECTION_HEADINGS = {
    "now": "NOTHING LOST · caches that rebuild themselves (deleted)",
    "stale": "REBUILD IF NEEDED · projects untouched for a while (to the Trash)",
    "models": "RE-DOWNLOAD IF NEEDED · local models (to the Trash; `disksage models` is gentler)",
    "docker": "DOCKER'S OWN CLEANUP",
    "judge": "YOURS TO JUDGE · to the Trash",
}


def _model_stores() -> set[str]:
    return {os.path.realpath(p) for p in models.model_dirs().values()}


def _bucket_of(it, stores: set[str]) -> str:
    if it.kind == "command":
        return "docker"
    if _deletes_directly(it):
        return "now"
    if _default_checked(it):
        return "stale"
    if it.kind == "dir":
        real = os.path.realpath(it.path)
        if any(real == s or real.startswith(s + os.sep) for s in stores):
            return "models"
    return "judge"


def _actionable(result: ScanResult) -> list:
    return [i for i in result.items if i.tier in ("safe", "duplicate", "review")]


def _buckets(result: ScanResult) -> tuple[list[Bucket], list]:
    """The summary lines, plus the items left alone (KEEP).

    Duplicate models live inside the model folders, so "nothing deleted"
    (dedupe) and "re-download" (deleting models) split those folders' bytes
    instead of counting them twice."""
    stores = _model_stores()
    groups: dict[str, list] = {"now": [], "stale": [], "models": [], "judge": []}
    kept = []
    for it in result.items:
        if it.tier == "keep":
            kept.append(it)
        elif it.tier in ("safe", "duplicate", "review"):
            groups[_bucket_of(it, stores)].append(it)

    def total(key):
        return sum(i.size for i in groups[key])

    buckets = []
    if groups["now"]:
        n = len(groups["now"])
        buckets.append(Bucket("now", total("now"), "nothing lost",
                              f"{n} cache{'s' if n != 1 else ''} that rebuild themselves"))
    dedupe = result.dupes.reclaimable if result.dupes else 0
    if dedupe:
        n = result.dupes.groups
        buckets.append(Bucket("dedupe", dedupe, "nothing deleted",
                              f"{n} model{'s' if n != 1 else ''} stored by more than one app"
                              f" · disksage dedupe", approx=not result.dupes.exact))
    if groups["stale"]:
        n = len(groups["stale"])
        projects = len({os.path.dirname(i.path) for i in groups["stale"]})
        buckets.append(Bucket("stale", total("stale"), "rebuild if needed",
                              f"{n} folder{'s' if n != 1 else ''} in {projects} "
                              f"project{'s' if projects != 1 else ''} untouched for "
                              f"{STALE_DAYS}+ days"))
    models_rest = max(0, total("models") - dedupe) if groups["models"] else 0
    if models_rest:
        buckets.append(Bucket("models", models_rest, "re-download if needed",
                              "the local models themselves · disksage models"))
    docker = result.docker.reclaimable if result.docker else 0
    if docker:
        buckets.append(Bucket("docker", docker, "Docker's own cleanup",
                              "unused images, containers and build cache"))
    if groups["judge"]:
        judge = sorted(groups["judge"], key=lambda i: -i.size)
        labels = list(dict.fromkeys(i.label for i in judge if i.label))
        examples = ", ".join(labels[:2]) + ("…" if len(labels) > 2 else "")
        n = len(judge)
        buckets.append(Bucket("judge", total("judge"), "yours to judge",
                              f"{n} other item{'s' if n != 1 else ''}"
                              + (f": {examples}" if examples else "")))
    return buckets, kept


def _item_notes(it) -> list[str]:
    """A short note for the biggest-items list, longest wording first."""
    if it.kind == "duplicate" and it.original:
        return [f"same as {_short_path(it.original)}", "duplicate"]
    if getattr(it, "tracked", 0):
        return ["tracked by git"]
    if it.touched and os.path.basename(it.path) in REBUILDABLE:
        return [f"last touched {_ago(it.touched)}", _ago(it.touched)]
    return []


def _path_and_note(it, room: int) -> tuple[str, str]:
    """Fit a path and its note into `room` columns. The path matters more (it
    says which project or cache), so the note gets shorter, then disappears,
    before the path is shortened."""
    path = _display_path(it)
    for note in _item_notes(it):
        if len(path) + len(note) + 3 <= room:
            return path, note
    return _fit(path, room), ""


def _print_summary(result: ScanResult, con, top: int = 8) -> None:
    buckets, kept = _buckets(result)
    total = sum(b.size for b in buckets)
    free = human(_free_bytes())
    if not buckets:
        _say(con, f"\n[bold]Nothing worth cleaning above the size threshold[/bold] "
                  f"({free} free). Try --min-size 5.",
             f"\nNothing worth cleaning above the size threshold ({free} free). "
             f"Try --min-size 5.")
    else:
        approx = "≈" if any(b.approx for b in buckets) else ""
        _say(con, f"\n[bold]Free up to {approx}{human(total)} on this Mac[/bold] · "
                  f"{free} free now\n",
             f"\nFree up to {approx}{human(total)} on this Mac · {free} free now\n")
        for b in buckets:
            style = BUCKET_STYLE.get(b.key, "white")
            size = ("≈" if b.approx else "") + human(b.size)
            _say(con, f"  {size:>10}  [{style}]{b.effect:<22}[/{style}] {b.what}",
                 f"  {size:>10}  {b.effect:<22} {b.what}")
        if any(b.approx for b in buckets):
            _say(con, "  [dim]≈ matched by size; `disksage dedupe` checks every file "
                      "before changing it[/dim]",
                 "  ≈ matched by size; `disksage dedupe` checks every file before changing it")
    if kept:
        n = len(kept)
        _say(con, f"\n  [dim]Left alone: {n} item{'s' if n != 1 else ''} of your data "
                  f"({human(sum(i.size for i in kept))})[/dim]",
             f"\n  Left alone: {n} item(s) of your data ({human(sum(i.size for i in kept))})")

    biggest = sorted(_actionable(result), key=lambda i: -i.size)[:top]
    if biggest:
        width = max(40, _columns())
        _say(con, "\n[bold]Biggest items[/bold]", "\nBiggest items")
        for it in biggest:
            label = it.label if len(it.label) <= 24 else it.label[:23] + "…"
            path, note = _path_and_note(it, max(24, width - 42))
            _say(con, f"  {human(it.size):>10}  {label:<24}  {path}"
                      + (f" [dim]· {note}[/dim]" if note else ""),
                 f"  {human(it.size):>10}  {label:<24}  {path}" + (f" · {note}" if note else ""))

    notes = []
    if result.suggest_setup and result.unknown_count:
        n = result.unknown_count
        notes.append(f"{n} folder{'s' if n != 1 else ''} ({human(result.unknown_bytes)}) aren't "
                     f"in the built-in list. `disksage setup` lets an AI identify them "
                     f"(it sees only paths and sizes).")
    if result.cloud_only_files:
        notes.append(f"Skipped {result.cloud_only_files} file(s) "
                     f"({human(result.cloud_only_bytes)}) that are only in iCloud or another "
                     f"cloud drive: deleting them wouldn't free space here.")
    tips = [c.display for c in tools.available() if c.tool != "docker"]
    if tips:
        notes.append("Your tools' own cleanups (they keep what you still use): "
                     + " · ".join(tips))
    if notes:
        print()
        for note in notes:
            _say(con, f"[dim]{note}[/dim]", note)


def _print_report(items, con) -> None:
    """Every item, as a table (`scan --details`)."""
    items = sorted(items, key=lambda i: (TIER_ORDER.get(i.tier, 3), -i.size))
    if con:
        from rich.table import Table
        table = Table(show_lines=False, expand=True)
        table.add_column("", width=6)
        table.add_column("Size", justify="right", width=9)
        table.add_column("What", width=22)
        table.add_column("Why / path", overflow="fold")
        for it in items:
            style, tag = TIER_STYLE[it.tier]
            src = "🤖" if it.source == "ai" else ""
            table.add_row(f"[{style}]{tag}[/{style}]", human(it.size),
                          f"{it.label} {src}".strip(),
                          f"{it.why}\n[dim]{_short_path(it.path)}[/dim]")
        con.print(table)
        con.print("[dim]🤖 = labelled by AI · SAFE rebuilds itself · REVIEW goes to the Trash"
                  " · KEEP is never touched[/dim]")
    else:
        for it in items:
            _, tag = TIER_STYLE[it.tier]
            print(f"{tag} {human(it.size):>9}  {it.label:22}  {_short_path(it.path)}")


def _print_docker_rows(report: tools.DockerReport) -> None:
    print(f"  {'':12}{'size':>12}{'reclaimable':>20}")
    for kind, size, reclaimable in report.rows:
        print(f"  {kind:<12}{size:>12}{reclaimable:>20}")


# ---- commands ----

def cmd_scan(args) -> int:
    con = _console()
    result = _scan(args, con)
    if args.details:
        _print_report(result.items, con)
        if result.docker:
            print("\nDocker (inside its own disk image):")
            _print_docker_rows(result.docker)
    _print_summary(result, con, top=0 if args.details else 8)
    more = "" if args.details else " · `--details` lists every item"
    _say(con, f"\n[bold]Nothing was changed.[/bold] Next: `disksage clean` to choose what "
              f"to remove{more}",
         f"\nNothing was changed. Next: `disksage clean` to choose what to remove{more}")
    return 0


def cmd_home(args) -> int:
    """`disksage` in a terminal: scan, summarize, then clean using the same scan."""
    con = _console()
    result = _scan(args, con)
    _print_summary(result, con)
    if not _actionable(result) and not (result.docker and result.docker.reclaimable):
        return 0
    if _ask("\nReview and clean now? [Y/n] ") in ("n", "no"):
        print("Nothing changed. `disksage clean` picks up from this scan for the next "
              f"{SCAN_REUSE_SECONDS // 60} minutes.")
        return 0
    return _clean(result, args, con)


def cmd_clean(args) -> int:
    con = _console()
    result = None if args.rescan else _load_scan(args)
    if result:
        _say(con, f"[dim]Using the scan from {_ago_short(result.created)} "
                  f"({len(result.items)} items). Add --rescan to scan again.[/dim]",
             f"Using the scan from {_ago_short(result.created)}. Add --rescan to scan again.")
    else:
        result = _scan(args, con)
    return _clean(result, args, con)


def _docker_item(result: ScanResult):
    report = result.docker
    if not report or report.reclaimable <= 0:
        return None
    return Item(path=DOCKER_ITEM, size=report.reclaimable, kind="command", tier="review",
                label="Docker cleanup", why=tools.DOCKER_PRUNE.removes, source="docker")


def _clean(result: ScanResult, args, con) -> int:
    actionable = _actionable(result)
    docker = _docker_item(result)
    if docker:
        actionable.append(docker)
    if not actionable:
        print("Nothing to clean.")
        return 0

    total = sum(i.size for i in actionable)
    _say(con, f"\n[bold]Choose what to clean[/bold] · {len(actionable)} items, up to "
              f"{human(total)}. Caches are deleted; everything else goes to the Trash.",
         f"\nChoose what to clean: {len(actionable)} items, up to {human(total)}.")
    queue = _select_tui(actionable) if _tui_available() else _select_prompts(actionable)
    if not queue:
        print("\nNothing selected. Nothing changed.")
        return 0

    # ---- final review + one explicit confirmation before anything moves ----
    deletes = [i for i in queue if _deletes_directly(i)]
    commands = [i for i in queue if i.kind == "command"]
    trashes = [i for i in queue if i not in deletes and i not in commands]
    print(f"\n{'─' * 60}")
    print(f"About to clean {len(queue)} item(s), {human(sum(i.size for i in queue))}:")
    risky = 0
    for heading, group in (
            ("Delete (caches that rebuild themselves):", deletes),
            ("Move to the Trash (you can get these back):", trashes),
            ("Run Docker's own cleanup:", commands)):
        if not group:
            continue
        print(f"\n  {heading}")
        for it in group:
            _, tag = TIER_STYLE[it.tier]
            line = f"    {tag} {human(it.size):>9}  {_display_path(it)}"
            risk = git_risk(it.path) if it.kind == "dir" else None
            if getattr(it, "tracked", 0):
                tracked = f"{it.tracked} file(s) tracked by git: source, not build output"
                risk = f"{risk}; {tracked}" if risk else tracked
            if risk:
                line += f"\n        ⚠️  {risk}"
                risky += 1
            print(line)
    if risky:
        print(f"\n  ⚠️  {risky} item(s) above carry a git warning — review them.")

    if getattr(args, "dry_run", False):
        print("\nDry run: nothing was deleted or moved.")
        return 0

    if _ask(f"\nType 'yes' to clean these {len(queue)} item(s): ") != "yes":
        print("Aborted. Nothing was deleted.")
        return 0

    free_before = _free_bytes()
    trashed = done = 0
    try:
        for n, it in enumerate(queue, 1):
            step = f"{n}/{len(queue)} · {_display_path(it)} ({human(it.size)})"
            if it.kind == "command":
                with _status(con, f"Running {tools.DOCKER_PRUNE.display}…"):
                    ok, last = tools.run(tools.DOCKER_PRUNE)
                print(f"  {'ran' if ok else 'FAILED'}: {tools.DOCKER_PRUNE.display}"
                      + (f" ({last})" if last else ""))
                done += 1
                continue
            ok, reason = is_safe_to_trash(it.path)   # safety floor, re-checked at execution
            if not ok:
                print(f"  skipped (protected): {_short_path(it.path)} — {reason}")
                continue
            try:
                if _deletes_directly(it):
                    with _status(con, f"Deleting {step}…"):
                        remove_regenerable(it.path)
                    print(f"  deleted: {_short_path(it.path)}")
                else:
                    with _status(con, f"Moving to the Trash {step}…"):
                        trash(it.path)
                    trashed += it.size
                    print(f"  trashed: {_short_path(it.path)}")
                done += 1
            except Exception as e:
                print(f"  FAILED: {_short_path(it.path)} — {e}")
    except KeyboardInterrupt:
        print(f"\nStopped after {done} of {len(queue)} item(s). Nothing else was changed.")
        _invalidate_scan()
        _report_freed(free_before, trashed, offer_empty=False)
        return 130

    _invalidate_scan()  # the disk changed; the next clean should scan again
    _report_freed(free_before, trashed)
    if result.dupes and result.dupes.reclaimable:
        print(f"Also: {human(result.dupes.reclaimable)} of duplicate models can be reclaimed "
              f"with `disksage dedupe` (nothing gets deleted).")
    return 0


def _ask(prompt: str) -> str:
    try:
        return input(prompt).strip().lower()
    except (EOFError, KeyboardInterrupt):
        print()
        return "n"


def _is_interactive() -> bool:
    return sys.stdin.isatty() and sys.stdout.isatty()


def _tui_available() -> bool:
    if not _is_interactive():
        return False
    try:
        import questionary  # noqa: F401
        return True
    except ImportError:
        return False


def _choice_title(it, path_budget: int) -> str:
    # Fixed-width columns so everything lines up regardless of tier/AI marker:
    #   [   size] [AI] [label..............]  path
    size = human(it.size).rjust(9)
    ai = "🤖" if it.source == "ai" else "  "
    label = (it.label if len(it.label) <= 22 else it.label[:21] + "…").ljust(22)
    return f"{size}  {ai} {label}  {_fit(_display_path(it), path_budget)}"


def _select_tui(actionable):
    import questionary
    from questionary import Choice, Separator, Style

    # columns consumed before the path: pointer+checkbox (~6) + size/ai/label (~40)
    path_budget = max(24, _columns() - 48)

    style = Style([
        ("qmark", "fg:#8b5cf6 bold"),
        ("question", "bold"),
        ("pointer", "fg:#06b6d4 bold"),
        ("highlighted", "fg:#06b6d4 bold"),   # row under the cursor
        ("selected", "fg:#22c55e bold"),      # ticked rows → green
        ("separator", "fg:#9ca3af"),
        ("instruction", "fg:#9ca3af"),
    ])

    # Same sections, in the same order, as the summary.
    stores = _model_stores()
    sections: dict[str, list] = {key: [] for key in SECTION_HEADINGS}
    for it in actionable:
        sections[_bucket_of(it, stores)].append(it)

    choices = []
    for key, heading in SECTION_HEADINGS.items():
        group = sorted(sections[key], key=lambda i: -i.size)
        if not group:
            continue
        if choices:
            choices.append(Separator(" "))
        choices.append(Separator(f"── {heading} · {human(sum(i.size for i in group))} ──"))
        for it in group:
            choices.append(Choice(title=_choice_title(it, path_budget), value=it,
                                  checked=_default_checked(it)))

    selected = questionary.checkbox(
        "Tick what to clean (the ticked items are the ones that cost you nothing to rebuild):",
        choices=choices,
        pointer="❯",
        instruction="(↑↓ move · space = tick · a = all · i = invert · enter = confirm)",
        style=style,
    ).ask()
    return selected or []


def _select_prompts(actionable):
    """Fallback selection: bulk prompts + per-item y/N (used when no TTY)."""
    queue = []
    safe_bulk = [i for i in actionable if _default_checked(i) and _deletes_directly(i)]
    dups = [i for i in actionable if i.tier == "duplicate"]
    rest = [i for i in actionable if i not in safe_bulk and i not in dups]

    if safe_bulk:
        total = sum(i.size for i in safe_bulk)
        if _ask(f"\nDelete all {len(safe_bulk)} known caches that rebuild themselves "
                f"({human(total)})? [y/N] ") == "y":
            queue.extend(safe_bulk)
    if dups:
        total = sum(i.size for i in dups)
        if _ask(f"\nMove {len(dups)} DUPLICATE extra copies ({human(total)}) to Trash? "
                f"One original of each is always kept. [y/N] ") == "y":
            queue.extend(dups)
    for it in rest:
        tag = it.tier.upper() + (" 🤖" if it.source == "ai" else "")
        verb = "Run this?" if it.kind == "command" else "Trash this?"
        if _ask(f"{tag}  {human(it.size):>9}  {it.label}\n  {_display_path(it)}\n"
                f"  {verb} [y/N] ") == "y":
            queue.append(it)
    return queue


def _report_freed(free_before: int, trashed: int, offer_empty: bool = True) -> None:
    """The only number worth trusting: how much free space actually changed."""
    gained = max(0, _free_bytes() - free_before)
    print(f"\nFree space went up by {human(gained)} (measured).")
    if trashed:
        print(f"{human(trashed)} is in the Trash and frees its space once the Trash is emptied. "
              f"To undo, open the Trash and drag items back.")
        if offer_empty and sys.platform == "darwin" and _ask(
                "Empty the Trash now? This also removes anything else already in it. [y/N] ") == "y":
            ok, err = empty_trash()
            if ok:
                gained = max(0, _free_bytes() - free_before)
                print(f"Trash emptied. Free space is up by {human(gained)} in total.")
            else:
                print(f"Couldn't empty the Trash ({err}). You can empty it from the Dock.")
    print(f"Disk: {_free_space()}")


# ---- local models: inventory and clone-based dedupe ----

def _duplicate_plan(min_size: int, con=None):
    """Every model file, plus (keeper, copies, copies-to-clone, reclaimable
    bytes) per duplicate group. Copies disksage already turned into clones of
    the same keeper don't count again."""
    files = models.all_model_files(min_size)
    records = models.load_clones()
    to_read = sum(f.size for f in models.files_to_hash(files))
    with _bytes_progress(con, to_read, "Checking model files") as advance:
        groups = models.find_model_duplicates(files, on_bytes=advance)
    plan = []
    for group in groups:
        keep = models.pick_keeper(group)
        copies = [f for f in group if f is not keep]
        todo = [f for f in copies if not models.already_shared(records, keep, f)]
        plan.append((keep, copies, todo, sum(f.on_disk for f in todo)))
    plan.sort(key=lambda p: -p[3])
    return files, plan, records


def _apps(files) -> str:
    return ", ".join(sorted({models.STORE_NAMES[f.store] for f in files}))


def cmd_models(args) -> int:
    con = None if args.json else _console()
    files, plan, _ = _duplicate_plan(args.min_size * MB, con)
    if args.json:
        print(json.dumps({
            "models": [{"app": f.store, "path": f.path, "bytes": f.size,
                        "bytes_on_disk": f.on_disk, "sha256": f.sha256, "names": list(f.names),
                        "file_name": f.file_name}
                       for f in files],
            "duplicates": [{"sha256": keep.sha256, "bytes": keep.size, "keep": keep.path,
                            "copies": [c.path for c in copies], "reclaimable_bytes": reclaim}
                           for keep, copies, _, reclaim in plan],
        }, indent=2))
        return 0

    if not files:
        print("No local models found (looked in Ollama, LM Studio and the Hugging Face cache).")
        return 0
    total = sum(f.on_disk for f in files)
    _say(con, f"[bold]{len(files)} model file(s), {human(total)} on disk[/bold]",
         f"{len(files)} model file(s), {human(total)} on disk")
    for f in files:
        print(f"  {human(f.on_disk):>9}  {models.STORE_NAMES[f.store]:<12}  {f.title}")
        if args.paths:
            _say(con, f"             [dim]{_short_path(f.path)}[/dim]",
                 f"             {_short_path(f.path)}")

    reclaimable = sum(p[3] for p in plan)
    if plan:
        _say(con, "\n[bold]The same file, stored by more than one app:[/bold]",
             "\nThe same file, stored by more than one app:")
        for keep, copies, todo, reclaim in plan:
            group = [keep, *copies]
            state = f"{human(reclaim)} reclaimable" if todo else "already shared (cloned)"
            print(f"  {keep.title} · {human(keep.size)} × {len(group)} copies "
                  f"({_apps(group)}) · {state}")
    if reclaimable:
        _say(con, f"\n[bold]{human(reclaimable)} reclaimable.[/bold] Run `disksage dedupe` to "
                  f"store each file once: every app keeps its file, nothing is deleted.",
             f"\n{human(reclaimable)} reclaimable. Run `disksage dedupe` to store each file once.")
    print("\nTo remove a model for good: "
          + "; ".join(f"{models.STORE_NAMES[s]}: {h}" for s, h in models.REMOVE_HINTS.items()))
    if not args.paths:
        print("Add --paths to see where each file is.")
    return 0


def cmd_dedupe(args) -> int:
    con = _console()
    _, plan, records = _duplicate_plan(args.min_size * MB, con)
    todo = [(keep, f) for keep, _, copies, _ in plan for f in copies]
    if not todo:
        print("No duplicate models to reclaim.")
        return 0
    total = sum(f.on_disk for _, f in todo)
    print(f"Replace {len(todo)} duplicate model file(s) with APFS clones: "
          f"up to {human(total)} reclaimable.")
    print("Every app keeps a working file at the same path; the bytes are stored once.\n")
    for keep, f in todo:
        print(f"  {human(f.on_disk):>9}  {models.STORE_NAMES[f.store]:<12}  {f.title}")
        _say(con, f"             [dim]becomes a clone of the {models.STORE_NAMES[keep.store]} "
                  f"copy[/dim]",
             f"             becomes a clone of the {models.STORE_NAMES[keep.store]} copy")
        if args.paths:
            _say(con, f"             [dim]{_short_path(f.path)}\n             → "
                      f"{_short_path(keep.path)}[/dim]",
                 f"             {_short_path(f.path)}\n             → {_short_path(keep.path)}")

    if getattr(args, "dry_run", False):
        print("\nDry run: nothing was changed.")
        return 0
    if sys.platform != "darwin":
        print("\nReclaiming duplicates needs macOS (APFS clones). Nothing was changed.")
        return 1
    if _ask(f"\nType 'yes' to clone these {len(todo)} file(s): ") != "yes":
        print("Aborted. Nothing was changed.")
        return 0

    free_before = _free_bytes()
    lines = []
    # Each pair is read in full (both files) to prove they're identical first.
    with _bytes_progress(con, sum(2 * f.size for _, f in todo), "Verifying and cloning") as advance:
        for keep, f in todo:
            ok, reason = is_safe_to_trash(f.path)   # same safety floor as `clean`
            if not ok:
                lines.append(f"  skipped (protected): {_short_path(f.path)} — {reason}")
                advance(2 * f.size)
                continue
            try:
                if models.replace_with_clone(keep.path, f.path, on_bytes=advance):
                    models.record_clone(records, keep.path, f.path)
                    lines.append(f"  cloned: {f.title} ({models.STORE_NAMES[f.store]})")
                else:
                    lines.append(f"  skipped (not identical, or on another volume): {f.title}")
            except Exception as e:
                lines.append(f"  FAILED: {_short_path(f.path)} — {e}")
    print("\n".join(lines))
    models.save_clones(records)
    _invalidate_scan()
    gained = max(0, _free_bytes() - free_before)
    print(f"\nDone. Free space went up by {human(gained)} (measured).")
    print(f"Disk: {_free_space()}")
    return 0


def _scan_json(args) -> int:
    """`scan --json`: the same scan, as data for scripts and agents."""
    with contextlib.redirect_stdout(io.StringIO()):   # keep progress text out of the JSON
        result = _scan(args, None)
    buckets, _ = _buckets(result)
    print(json.dumps({
        "version": __version__,
        "free_bytes": _free_bytes(),
        "files_seen": result.files_seen,
        "seconds": round(result.seconds, 2),
        "summary": [{"group": b.key, "bytes": b.size, "effect": b.effect, "what": b.what,
                     "estimate": b.approx}
                    for b in buckets],
        "items": [{"path": i.path, "tier": i.tier, "label": i.label, "why": i.why,
                   "source": i.source, "kind": i.kind, "bytes_on_disk": i.size,
                   "pre_ticked": _default_checked(i), "git_tracked_files": i.tracked,
                   **({"original": i.original} if i.original else {})}
                  for i in sorted(result.items, key=lambda i: -i.size)],
    }, indent=2))
    return 0


def main(argv=None) -> int:
    p = argparse.ArgumentParser(
        prog="disksage",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        description="AI-aware disk cleaner that never deletes without asking.",
        epilog=(
            "examples:\n"
            "  disksage              scan, see what can be freed, then choose what to clean\n"
            "  disksage scan         read-only report (--details for every item)\n"
            "  disksage clean        choose what to clean from a checklist\n"
            "  disksage clean --dry-run   show what clean would do, change nothing\n"
            "  disksage models       every local model, and the ones stored twice\n"
            "  disksage dedupe       reclaim duplicate models as APFS clones\n"
            "  disksage setup        choose an AI provider / paste a key\n"
        ))
    p.add_argument("--version", action="version", version=f"disksage {__version__}")
    sub = p.add_subparsers(dest="cmd", metavar="{scan,clean,models,dedupe,setup,doctor}")

    sub.add_parser("setup", help="choose an AI provider and save your key")
    sub.add_parser("doctor", help="show the active provider and detected local servers")

    mp = sub.add_parser("models", help="list local models (Ollama, LM Studio, Hugging Face) "
                                       "and find the ones stored more than once")
    mp.add_argument("--min-size", type=int, default=50, help="MB threshold (default 50)")
    mp.add_argument("--json", action="store_true", help="print the result as JSON")
    mp.add_argument("--paths", action="store_true", help="show where each file is")
    dp = sub.add_parser("dedupe", help="replace duplicate model files with APFS clones")
    dp.add_argument("--min-size", type=int, default=50, help="MB threshold (default 50)")
    dp.add_argument("--dry-run", action="store_true", help="show the plan, change nothing")
    dp.add_argument("--paths", action="store_true", help="show the paths of both copies")

    for name in ("scan", "clean"):
        sp = sub.add_parser(name)
        sp.add_argument("--min-size", type=int, default=20,
                        help="MB threshold for folders (default 20)")
        sp.add_argument("--big-size", type=int, default=200,
                        help="MB threshold for loose files (default 200)")
        sp.add_argument("--dup-size", type=int, default=50,
                        help="MB threshold for duplicate detection (default 50)")
        sp.add_argument("--no-files", action="store_true",
                        help="skip big-file and duplicate scanning")
        sp.add_argument("--provider", default=None, help="local|groq|gemini")
        sp.add_argument("--no-ai", action="store_true",
                        help="use the built-in list only, never call an AI")
        sp.add_argument("--root", default=None,
                        help="scan only this folder (project folders and big files)")
        if name == "clean":
            sp.add_argument("--dry-run", action="store_true",
                            help="show what would be deleted or trashed, change nothing")
            sp.add_argument("--rescan", action="store_true",
                            help="scan again instead of reusing a scan from the last "
                                 f"{SCAN_REUSE_SECONDS // 60} minutes")
        else:
            sp.add_argument("--details", action="store_true",
                            help="list every item, not just the biggest")
            sp.add_argument("--json", action="store_true",
                            help="print the result as JSON (for scripts and agents)")

    args = p.parse_args(argv)
    config.load()
    try:
        return _run(p, args)
    except KeyboardInterrupt:
        print("\nStopped. Nothing else was changed.")
        return 130


def _run(p, args) -> int:
    if args.cmd == "setup":
        return cmd_setup(args)
    if args.cmd == "doctor":
        return cmd_doctor(args)
    if args.cmd == "models":
        return cmd_models(args)
    if args.cmd == "dedupe":
        return cmd_dedupe(args)

    if getattr(args, "provider", None):
        os.environ["DISKSAGE_PROVIDER"] = args.provider

    # No subcommand: in a terminal, the whole flow (scan → summary → clean);
    # when piped, the read-only report.
    bare = args.cmd is None
    if bare:
        args = p.parse_args(["scan"])
        if _is_interactive():
            return cmd_home(args)

    # No setup questions before the first result: the built-in list works on its
    # own, and the summary says when an AI could identify more.
    if args.cmd == "clean":
        return cmd_clean(args)
    if getattr(args, "json", False):
        return _scan_json(args)
    return cmd_scan(args)


if __name__ == "__main__":
    sys.exit(main())
