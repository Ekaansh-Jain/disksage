"""Command line interface.

    disksage            scan and report (default) — deletes nothing
    disksage scan       same as above
    disksage clean      scan, then pick items from a checklist to remove: caches
                       that rebuild themselves are deleted, everything else goes
                       to the Trash (per-item y/N when input isn't a terminal)

Flags:
    --min-size MB      ignore items smaller than this (default 20)
    --provider NAME    force local | groq | gemini
    --no-ai            knowledge base only, no LLM
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
import time
from contextlib import contextmanager

from . import __version__, config, llm, models, tools
from .classify import annotate_known, annotate_with_ai, build_file_items, human
from .safety import (
    empty_trash,
    git_risk,
    is_safe_to_trash,
    remove_regenerable,
    tracked_file_count,
    trash,
)
from .scan import CRUFT_DIR_NAMES, collect_files, project_last_touched, scan_dirs

HOME = os.path.expanduser("~")

TIER_STYLE = {
    "safe":      ("green",     "SAFE  "),
    "duplicate": ("magenta",   "DUP   "),
    "review":    ("yellow",    "REVIEW"),
    "keep":      ("red",       "KEEP  "),
}
TIER_ORDER = {"safe": 0, "duplicate": 1, "review": 2, "keep": 3}


def _console():
    try:
        from rich.console import Console
        return Console()
    except ImportError:
        return None


def _free_space() -> str:
    usage = shutil.disk_usage("/")
    return f"{human(usage.free)} free of {human(usage.total)}"


def _print_report(items, con) -> None:
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
                          f"{it.why}\n[dim]{it.path}[/dim]")
        con.print(table)
    else:
        for it in items:
            _, tag = TIER_STYLE[it.tier]
            print(f"{tag} {human(it.size):>9}  {it.label:22}  {it.path}")


def _totals(items) -> dict[str, int]:
    out = {"safe": 0, "duplicate": 0, "review": 0, "keep": 0}
    for it in items:
        out[it.tier] = out.get(it.tier, 0) + it.size
    return out


def _custom_local(questionary) -> dict:
    base = questionary.text(
        "Base URL of the OpenAI-compatible server "
        "(e.g. http://localhost:11434/v1):").ask()
    if not base or not base.strip():
        return {}
    base = base.strip()
    if not base.startswith(("http://", "https://")):
        base = "http://" + base
    models = llm.list_models(base)
    if models:
        model = models[0] if len(models) == 1 else (
            questionary.select("Which model?", choices=models).ask() or models[0])
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
    print(f"\nSaved to {path}")
    print(f"AI provider: {llm.describe()}")
    if not first_run:
        print("\nNext:  disksage scan   (report where space went)")
        print("       disksage clean  (pick items to free up)")
    return 0


def _print_provider_banner(con) -> None:
    cfg = llm.resolve()
    if cfg:
        msg = (f"AI: {cfg['name']} ({cfg['model']}) — only file paths + sizes "
               f"are sent for labelling, never file contents.")
        con.print(f"[dim]{msg}[/dim]") if con else print(msg)
        return
    msg = "No AI configured — using the built-in list only."
    con.print(f"[dim]{msg}[/dim]") if con else print(msg)


def _say(con, rich_msg: str, plain_msg: str) -> None:
    con.print(rich_msg) if con else print(plain_msg)


@contextmanager
def _status(con, text: str):
    """Show a spinner (rich) or a plain line while a slow step runs."""
    if con:
        with con.status(text):
            yield
    else:
        print(text)
        yield


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
      projects get their rebuildable folders pre-ticked."""
    for it in items:
        name = os.path.basename(it.path)
        if it.kind != "dir" or name not in CRUFT_DIR_NAMES:
            continue
        it.tracked = tracked_file_count(it.path)
        if it.tracked:
            it.tier = "review"
            it.why = (f"⚠ {it.tracked} file(s) here are tracked by git: "
                      f"this is source, not build output")
            continue
        it.touched = project_last_touched(it.path)
        if it.touched and name in REBUILDABLE:
            it.why = f"{it.why}; project last touched {_ago(it.touched)}"


def _scan_and_classify(args, con):
    mb = 1024 * 1024
    _print_provider_banner(con)

    with _status(con, "Scanning disk…"):
        dir_items = scan_dirs(min_size=args.min_size * mb, root=args.root)

    unknown = annotate_known(dir_items)
    _flag_project_folders(dir_items)
    provider = llm.resolve()
    if not args.no_ai and unknown and not provider:
        # Results first; the AI is an optional extra, offered once we know it helps.
        size = human(sum(u.size for u in unknown))
        _say(con,
             f"[yellow]{len(unknown)} folder(s) ({size}) aren't in the built-in list and stay "
             f"REVIEW.[/yellow] [dim]Run `disksage setup` to let an AI identify them "
             f"(it only sees paths and sizes).[/dim]",
             f"{len(unknown)} folder(s) ({size}) aren't in the built-in list. "
             f"Run `disksage setup` to let an AI identify them.")
    if not args.no_ai and unknown and provider:
        with _status(con, f"Identifying {len(unknown)} unrecognized item(s) with {provider['name']}…"):
            outcome = annotate_with_ai(unknown)
        if outcome.ok:
            _say(con,
                 f"[green]✓ Identified {outcome.labeled}/{outcome.total} unknown item(s) with {provider['name']}.[/green]",
                 f"Identified {outcome.labeled}/{outcome.total} unknown items with {provider['name']}.")
        else:
            _say(con,
                 f"[yellow]⚠ AI unavailable — {outcome.message}.\n"
                 f"  Falling back to the knowledge base; {outcome.total} folder(s) stay labelled 'review'.[/yellow]",
                 f"AI unavailable — {outcome.message}. "
                 f"Falling back to the knowledge base; {outcome.total} folders stay labelled 'review'.")

    file_items = []
    if not args.no_files:
        roots = [args.root] if args.root else None
        stats: dict = {}
        with _status(con, "Checking big files & duplicates…"):
            files = collect_files(file_min=args.dup_size * mb, roots=roots, stats=stats)
            file_items = build_file_items(files, big_min=args.big_size * mb)
        if stats.get("cloud_only_files"):
            # Placeholders free nothing locally, and reading them would download them.
            _say(con,
                 f"[dim]Skipped {stats['cloud_only_files']} file(s) "
                 f"({human(stats['cloud_only_bytes'])}) that are only in iCloud or another "
                 f"cloud drive; deleting them wouldn't free space on this Mac.[/dim]",
                 f"Skipped {stats['cloud_only_files']} cloud-only file(s) "
                 f"({human(stats['cloud_only_bytes'])}).")

    return dir_items + file_items


def cmd_scan(args) -> int:
    con = _console()
    items = _scan_and_classify(args, con)
    if not items:
        print("Nothing above the size threshold. Try --min-size 5.")
        return 0
    _print_report(items, con)

    legend = ("Legend  🤖 = labelled by AI (others from the built-in list)   ·   "
              "SAFE rebuilds itself   ·   REVIEW check first (goes to the Trash)   ·   "
              "KEEP left alone")
    con.print(f"[dim]{legend}[/dim]") if con else print(legend)

    t = _totals(items)
    reclaimable = t["safe"] + t["duplicate"] + t["review"]
    non_cache = t["duplicate"] + t["review"]
    if reclaimable > 0 and non_cache > 0:
        headline = (f"\n{human(reclaimable)} reclaimable — only {human(t['safe'])} is cache.\n"
                    f"The rest is dev cruft, duplicates & large files a cache cleaner misses.")
        con.print(f"[bold]{headline}[/bold]") if con else print(headline)
    msg = (f"\nBreakdown → cache: {human(t['safe'])}   "
           f"duplicates: {human(t['duplicate'])}   review: {human(t['review'])}   "
           f"(keep: {human(t['keep'])})\n"
           f"Disk: {_free_space()}")
    con.print(msg) if con else print(msg)

    _print_tool_tips(con)
    _print_model_hint(con)

    hint = ("\nThis was a read-only report — nothing was deleted.\n"
            "→ Run `disksage clean` to pick items from a checklist and free space.")
    con.print(f"[bold]{hint}[/bold]") if con else print(hint)
    return 0


def _print_model_hint(con) -> None:
    """Point at `disksage models` when two apps hold model files of exactly the
    same size. Only sizes are compared here (no hashing), so scan stays fast."""
    sizes: dict[int, set[str]] = {}
    for f in models.all_model_files():
        sizes.setdefault(f.size, set()).add(f.store)
    if any(len(stores) > 1 for stores in sizes.values()):
        _say(con, "\n[bold]Some models look like they're stored by more than one app.[/bold] "
                  "Run `disksage models` to check, and `disksage dedupe` to store each once.",
             "\nSome models look like they're stored by more than one app. "
             "Run `disksage models` to check.")


def _print_tool_tips(con) -> None:
    """Space a file scan can't judge: Docker's disk image, and tools that can
    clean up after themselves while keeping what's still in use."""
    usage = tools.docker_usage()
    if usage:
        _say(con, "\n[bold]Docker[/bold] (inside its own disk image; `disksage clean` "
                  "can run Docker's cleanup for you):",
             "\nDocker (inside its own disk image; `disksage clean` can run Docker's cleanup):")
        print("\n".join(f"  {line}" for line in usage.splitlines()))
    tips = [c for c in tools.available() if c.tool != "docker"]
    if tips:
        _say(con, "\n[bold]Your tools' own cleanup commands[/bold] "
                  "(they keep what you still use; disksage doesn't run these):",
             "\nYour tools' own cleanup commands (disksage doesn't run these):")
        for c in tips:
            print(f"  {c.display:<34} {c.removes}")


def _ask(prompt: str) -> str:
    try:
        return input(prompt).strip().lower()
    except (EOFError, KeyboardInterrupt):
        print()
        return "n"


def _tui_available() -> bool:
    if not (sys.stdin.isatty() and sys.stdout.isatty()):
        return False
    try:
        import questionary  # noqa: F401
        return True
    except ImportError:
        return False


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


def _short_path(path: str) -> str:
    return "~" + path[len(HOME):] if path.startswith(HOME) else path


_TIER_HEADINGS = {
    "safe": "SAFE — rebuilds itself; deleted directly",
    "duplicate": "DUPLICATES — one original is always kept; tick the copies you don't need",
    "review": "REVIEW — reclaimable, but check first",
}


def _fit_path(path: str, budget: int) -> str:
    """Shorten a path to `budget` columns, keeping the meaningful tail
    (filename / deepest folders) and dropping the middle."""
    p = _short_path(path)
    if len(p) <= budget:
        return p
    return "…" + p[-(budget - 1):]


def _choice_title(it, path_budget: int) -> str:
    # Fixed-width columns so everything lines up regardless of tier/AI marker:
    #   [   size] [AI] [label..............]  path
    size = human(it.size).rjust(9)
    ai = "🤖" if it.source == "ai" else "  "
    label = (it.label if len(it.label) <= 22 else it.label[:21] + "…").ljust(22)
    return f"{size}  {ai} {label}  {_fit_path(it.path, path_budget)}"


def _select_tui(actionable):
    import questionary
    from questionary import Choice, Separator, Style

    cols = shutil.get_terminal_size((100, 24)).columns
    # columns consumed before the path: pointer+checkbox (~6) + size/ai/label (~40)
    path_budget = max(24, cols - 48)

    style = Style([
        ("qmark", "fg:#8b5cf6 bold"),
        ("question", "bold"),
        ("pointer", "fg:#06b6d4 bold"),
        ("highlighted", "fg:#06b6d4 bold"),   # row under the cursor
        ("selected", "fg:#22c55e bold"),      # ticked rows → green
        ("separator", "fg:#9ca3af"),
        ("instruction", "fg:#9ca3af"),
    ])

    ordered = sorted(actionable, key=lambda i: (TIER_ORDER.get(i.tier, 3), -i.size))
    tier_totals: dict[str, int] = {}
    for it in ordered:
        tier_totals[it.tier] = tier_totals.get(it.tier, 0) + it.size

    choices = []
    last_tier = None
    for it in ordered:
        if it.tier != last_tier:
            if last_tier is not None:
                choices.append(Separator(" "))
            heading = _TIER_HEADINGS.get(it.tier, it.tier.upper())
            choices.append(Separator(f"── {heading}  ·  {human(tier_totals[it.tier])} ──"))
            last_tier = it.tier
        choices.append(Choice(title=_choice_title(it, path_budget), value=it,
                              checked=_default_checked(it)))

    grand = human(sum(i.size for i in ordered))
    selected = questionary.checkbox(
        f"Select what to move to Trash  ·  {len(ordered)} items, up to {grand} reclaimable:",
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
        if _ask(f"{tag}  {human(it.size):>9}  {it.label}\n  {it.path}\n  Trash this? [y/N] ") == "y":
            queue.append(it)
    return queue


def cmd_clean(args) -> int:
    con = _console()
    intro = ("disksage clean — you'll get a checklist to tick what to remove.\n"
             "Nothing changes until you review the final list and type 'yes'. Caches that "
             "rebuild themselves are deleted; everything else goes to the Trash (recoverable).")
    con.print(f"[bold]{intro}[/bold]\n") if con else print(intro + "\n")

    items = _scan_and_classify(args, con)
    actionable = [i for i in items if i.tier in ("safe", "duplicate", "review")]

    if not actionable:
        print("Nothing reclaimable found.")
        return 0

    _print_report(items, con)
    print(f"\nDisk: {_free_space()}")

    if _tui_available():
        queue = _select_tui(actionable)
    else:
        queue = _select_prompts(actionable)

    if not queue:
        print("\nNothing selected. Nothing deleted.")
        return 0

    # ---- Step 2: final review + one explicit confirmation before anything moves ----
    deletes = [i for i in queue if _deletes_directly(i)]
    trashes = [i for i in queue if not _deletes_directly(i)]
    total = sum(i.size for i in queue)
    print(f"\n{'─' * 60}")
    print(f"About to clean {len(queue)} item(s) — {human(total)}:")
    risky = 0
    for heading, group in (
            ("Delete (caches that rebuild themselves):", deletes),
            ("Move to the Trash (you can get these back):", trashes)):
        if not group:
            continue
        print(f"\n  {heading}")
        for it in group:
            _, tag = TIER_STYLE[it.tier]
            line = f"    {tag} {human(it.size):>9}  {it.path}"
            risk = git_risk(it.path) if it.kind != "duplicate" else None
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

    confirm = _ask(f"\nType 'yes' to clean these {len(queue)} item(s): ")
    if confirm != "yes":
        print("Aborted. Nothing was deleted.")
        return 0

    free_before = _free_bytes()
    trashed = 0
    for it in queue:
        ok, reason = is_safe_to_trash(it.path)   # safety floor, re-checked at execution
        if not ok:
            print(f"  skipped (protected): {it.path} — {reason}")
            continue
        try:
            if _deletes_directly(it):
                remove_regenerable(it.path)
                print(f"  deleted: {it.path}")
            else:
                trash(it.path)
                trashed += it.size
                print(f"  trashed: {it.path}")
        except Exception as e:
            print(f"  FAILED: {it.path} — {e}")

    _offer_docker_prune()
    _report_freed(free_before, trashed)
    return 0


def _free_bytes() -> int:
    return shutil.disk_usage(HOME).free


def _deletes_directly(it) -> bool:
    """Caches that rebuild themselves (SAFE in the built-in list) are deleted
    outright: in the Trash they'd free nothing until it's emptied. Anything
    that could be yours (REVIEW, duplicates, AI-labelled items) goes to the
    Trash instead."""
    return it.tier == "safe" and it.source == "knowledge" and not getattr(it, "tracked", 0)


def _offer_docker_prune() -> None:
    """Docker's disk image is invisible to a file scan, so show Docker's own
    report and offer Docker's own cleanup command."""
    usage = tools.docker_usage()
    if not usage:
        return
    print("\nDocker keeps images and containers inside its own disk image. Docker's report:")
    print("\n".join(f"  {line}" for line in usage.splitlines()))
    prune = tools.DOCKER_PRUNE
    if _ask(f"Run `{prune.display}` to remove {prune.removes}? [y/N] ") == "y":
        ok, last = tools.run(prune)
        print(f"  {'done' if ok else 'failed'}{': ' + last if last else ''}")


def _report_freed(free_before: int, trashed: int) -> None:
    """The only number worth trusting: how much free space actually changed."""
    gained = max(0, _free_bytes() - free_before)
    print(f"\nDone. Free space went up by {human(gained)} (measured).")
    if trashed:
        print(f"{human(trashed)} is in the Trash and frees its space once the Trash is emptied.")
        if sys.platform == "darwin" and _ask(
                "Empty the Trash now? This also removes anything else already in it. [y/N] ") == "y":
            ok, err = empty_trash()
            if ok:
                gained = max(0, _free_bytes() - free_before)
                print(f"Trash emptied. Free space is up by {human(gained)} in total.")
            else:
                print(f"Couldn't empty the Trash ({err}). You can empty it from the Dock.")
    print(f"Disk: {_free_space()}")


# ---- local models: inventory and clone-based dedupe ----

def _duplicate_plan(min_size: int):
    """Every model file, plus (keeper, copies-to-clone, reclaimable bytes) per
    duplicate group. Copies disksage already turned into clones of the same
    keeper don't count again."""
    files = models.all_model_files(min_size)
    records = models.load_clones()
    plan = []
    for group in models.find_model_duplicates(files):
        keep = models.pick_keeper(group)
        copies = [f for f in group if f is not keep]
        todo = [f for f in copies if not models.already_shared(records, keep, f)]
        plan.append((keep, copies, todo, sum(f.on_disk for f in todo)))
    plan.sort(key=lambda p: -p[3])
    return files, plan, records


def _apps(files) -> str:
    return ", ".join(sorted({models.STORE_NAMES[f.store] for f in files}))


def _model_name(f) -> str:
    return f.names[0] if f.names else os.path.basename(f.path)


def cmd_models(args) -> int:
    files, plan, _ = _duplicate_plan(args.min_size * 1024 * 1024)
    if args.json:
        print(json.dumps({
            "models": [{"app": f.store, "path": f.path, "bytes": f.size,
                        "bytes_on_disk": f.on_disk, "sha256": f.sha256, "names": list(f.names)}
                       for f in files],
            "duplicates": [{"sha256": keep.sha256, "bytes": keep.size, "keep": keep.path,
                            "copies": [c.path for c in copies], "reclaimable_bytes": reclaim}
                           for keep, copies, _, reclaim in plan],
        }, indent=2))
        return 0

    con = _console()
    if not files:
        print("No local models found (looked in Ollama, LM Studio and the Hugging Face cache).")
        return 0
    total = sum(f.on_disk for f in files)
    _say(con, f"[bold]{len(files)} model file(s), {human(total)} on disk[/bold]",
         f"{len(files)} model file(s), {human(total)} on disk")
    for f in files:
        names = ", ".join(f.names) or "(not referenced by any model)"
        print(f"  {human(f.on_disk):>9}  {models.STORE_NAMES[f.store]:<12}  {names}")
        _say(con, f"             [dim]{_short_path(f.path)}[/dim]",
             f"             {_short_path(f.path)}")

    reclaimable = sum(p[3] for p in plan)
    if plan:
        _say(con, "\n[bold]The same file, stored by more than one app:[/bold]",
             "\nThe same file, stored by more than one app:")
        for keep, copies, todo, reclaim in plan:
            group = [keep, *copies]
            state = f"{human(reclaim)} reclaimable" if todo else "already shared (cloned)"
            print(f"  {_model_name(keep)} · {human(keep.size)} × {len(group)} copies "
                  f"({_apps(group)}) · {state}")
    if reclaimable:
        _say(con, f"\n[bold]{human(reclaimable)} reclaimable.[/bold] Run `disksage dedupe` to "
                  f"store each file once: every app keeps its file, nothing is deleted.",
             f"\n{human(reclaimable)} reclaimable. Run `disksage dedupe` to store each file once.")
    print("\nTo remove a model for good: "
          + "; ".join(f"{models.STORE_NAMES[s]}: {h}" for s, h in models.REMOVE_HINTS.items()))
    return 0


def cmd_dedupe(args) -> int:
    con = _console()
    _, plan, records = _duplicate_plan(args.min_size * 1024 * 1024)
    todo = [(keep, f) for keep, _, copies, _ in plan for f in copies]
    if not todo:
        print("No duplicate models to reclaim.")
        return 0
    total = sum(f.on_disk for _, f in todo)
    print(f"Replace {len(todo)} duplicate model file(s) with APFS clones — "
          f"up to {human(total)} reclaimable.")
    print("Every app keeps a working file at the same path; the bytes are stored once.\n")
    for keep, f in todo:
        print(f"  {human(f.on_disk):>9}  {models.STORE_NAMES[f.store]:<12} {_short_path(f.path)}")
        _say(con, f"             [dim]becomes a clone of {_short_path(keep.path)}[/dim]",
             f"             becomes a clone of {_short_path(keep.path)}")

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
    for keep, f in todo:
        ok, reason = is_safe_to_trash(f.path)   # same safety floor as `clean`
        if not ok:
            print(f"  skipped (protected): {f.path} — {reason}")
            continue
        try:
            if models.replace_with_clone(keep.path, f.path):
                models.record_clone(records, keep.path, f.path)
                print(f"  cloned: {_short_path(f.path)}")
            else:
                print(f"  skipped (not identical, or on another volume): {_short_path(f.path)}")
        except Exception as e:
            print(f"  FAILED: {f.path} — {e}")
    models.save_clones(records)
    gained = max(0, _free_bytes() - free_before)
    print(f"\nDone. Free space went up by {human(gained)} (measured).")
    print(f"Disk: {_free_space()}")
    return 0


def _scan_json(args) -> int:
    """`scan --json`: the same scan, as data for scripts and agents."""
    with contextlib.redirect_stdout(io.StringIO()):   # keep progress text out of the JSON
        items = _scan_and_classify(args, None)
    print(json.dumps({
        "version": __version__,
        "free_bytes": _free_bytes(),
        "items": [{"path": i.path, "tier": i.tier, "label": i.label, "why": i.why,
                   "source": i.source, "kind": i.kind, "bytes_on_disk": i.size,
                   "pre_ticked": _default_checked(i), "git_tracked_files": i.tracked}
                  for i in sorted(items, key=lambda i: -i.size)],
    }, indent=2))
    return 0


def main(argv=None) -> int:
    p = argparse.ArgumentParser(
        prog="disksage",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        description="AI-aware disk cleaner that never deletes without asking.",
        epilog=(
            "examples:\n"
            "  disksage              scan and report (deletes nothing)\n"
            "  disksage scan         same as above\n"
            "  disksage clean        pick items from a checklist and free space\n"
            "  disksage clean --dry-run   show what clean would do, change nothing\n"
            "  disksage models       every local model, and the ones stored twice\n"
            "  disksage dedupe       reclaim duplicate models as APFS clones\n"
            "  disksage setup        choose an AI provider / paste a key\n"
            "  disksage clean --min-size 5 --no-ai\n"
        ))
    p.add_argument("--version", action="version", version=f"disksage {__version__}")
    sub = p.add_subparsers(dest="cmd", metavar="{scan,clean,models,dedupe,setup,doctor}")

    sub.add_parser("setup", help="choose an AI provider and save your key")
    sub.add_parser("doctor", help="show the active provider and detected local servers")

    mp = sub.add_parser("models", help="list local models (Ollama, LM Studio, Hugging Face) "
                                       "and find the ones stored more than once")
    mp.add_argument("--min-size", type=int, default=50, help="MB threshold (default 50)")
    mp.add_argument("--json", action="store_true", help="print the result as JSON")
    dp = sub.add_parser("dedupe", help="replace duplicate model files with APFS clones")
    dp.add_argument("--min-size", type=int, default=50, help="MB threshold (default 50)")
    dp.add_argument("--dry-run", action="store_true", help="show the plan, change nothing")

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
        else:
            sp.add_argument("--json", action="store_true",
                            help="print the result as JSON (for scripts and agents)")

    args = p.parse_args(argv)
    config.load()

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

    # No subcommand → behave like `scan`.
    if not hasattr(args, "min_size"):
        args = p.parse_args(["scan"])

    # No setup questions before the first result: the built-in list works on its
    # own, and the scan says when an AI could identify more.
    if args.cmd == "clean":
        return cmd_clean(args)
    if getattr(args, "json", False):
        return _scan_json(args)
    return cmd_scan(args)


if __name__ == "__main__":
    sys.exit(main())
