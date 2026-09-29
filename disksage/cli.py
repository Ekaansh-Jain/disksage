"""Command line interface.

    disksage            scan and report (default) — deletes nothing
    disksage scan       same as above
    disksage clean      scan, then pick items from a checklist to move to Trash
                       (falls back to per-item y/N when input isn't a terminal)

Flags:
    --min-size MB      ignore items smaller than this (default 20)
    --provider NAME    force local | groq | gemini
    --no-ai            knowledge base only, no LLM
    --root PATH        (reserved) restrict scanning
"""

from __future__ import annotations

import argparse
import os
import shutil
import sys

from contextlib import contextmanager

HOME = os.path.expanduser("~")

from . import __version__, config, llm
from .classify import annotate_known, annotate_with_ai, build_file_items, human
from .safety import git_risk, is_safe_to_trash, trash
from .scan import collect_files, scan_dirs

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
            "Local LM Studio — private, no key (needs the app running)",
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
            values = {"GROQ_API_KEY": key.strip(),
                      "DISKSAGE_GROQ_MODEL": "openai/gpt-oss-20b"}
    elif choice.startswith("Google"):
        key = questionary.password(
            "Paste your Gemini API key (free at aistudio.google.com):").ask()
        if key and key.strip():
            values = {"GEMINI_API_KEY": key.strip(),
                      "DISKSAGE_GEMINI_MODEL": "gemini-flash-latest"}
    elif choice.startswith("Local"):
        values = {"DISKSAGE_PROVIDER": "local"}
        print("→ Start LM Studio's local server (Developer tab) before running disksage.")

    if not values:
        # "No AI", or a key prompt left blank — record the choice so we don't ask again.
        values = {"DISKSAGE_AI": "off"}

    path = config.save(values)
    print(f"\nSaved to {path}")
    print(f"AI provider: {llm.describe()}")
    if not first_run:
        print("\nNext:  disksage scan   (report where space went)")
        print("       disksage clean  (pick items to free up)")
    return 0


def _maybe_first_run_setup(args) -> None:
    """On the very first interactive run, offer setup once, then continue into
    the scan the user asked for (fastest path to value)."""
    if config.is_configured() or getattr(args, "no_ai", False):
        return
    if not sys.stdin.isatty():
        return
    print("👋 Welcome to disksage — quick one-time setup.\n")
    cmd_setup(first_run=True)
    config.load()
    print("\nSetup complete — running your first scan now.")
    print("(next time: `disksage scan` to report, `disksage clean` to free space)\n")


def _print_provider_banner(con) -> None:
    cfg = llm.resolve()
    if cfg:
        msg = (f"AI: {cfg['name']} ({cfg['model']}) — only file paths + sizes "
               f"are sent for labelling, never file contents.")
        con.print(f"[dim]{msg}[/dim]") if con else print(msg)
        return
    head = "No AI configured — running with the knowledge base only."
    tail = [
        "Known caches and junk are still found; unrecognized folders just stay",
        "labelled 'review' instead of being identified for you.",
        "To enable smarter labels (optional): add GROQ_API_KEY or GEMINI_API_KEY",
        "to a .env file, or run LM Studio locally. See the README.",
    ]
    if con:
        con.print(f"[yellow]{head}[/yellow]")
        for line in tail:
            con.print(f"[dim]{line}[/dim]")
    else:
        print(head)
        for line in tail:
            print(line)


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


def _scan_and_classify(args, con):
    mb = 1024 * 1024
    _print_provider_banner(con)

    with _status(con, "Scanning disk…"):
        dir_items = scan_dirs(min_size=args.min_size * mb, root=args.root)

    unknown = annotate_known(dir_items)
    provider = llm.resolve()
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
        with _status(con, "Checking big files & duplicates…"):
            files = collect_files(file_min=args.dup_size * mb, roots=roots)
            file_items = build_file_items(files, big_min=args.big_size * mb)

    return dir_items + file_items


def cmd_scan(args) -> int:
    con = _console()
    items = _scan_and_classify(args, con)
    if not items:
        print("Nothing above the size threshold. Try --min-size 5.")
        return 0
    _print_report(items, con)
    t = _totals(items)
    msg = (f"\nReclaimable now → safe: {human(t['safe'])}   "
           f"duplicates: {human(t['duplicate'])}   review: {human(t['review'])}   "
           f"(keep: {human(t['keep'])})\n"
           f"Disk: {_free_space()}")
    con.print(msg) if con else print(msg)
    print("\nRun `disksage clean` to move approved items to the Trash (reversible).")
    return 0


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
    # Pre-tick only the low-risk items: knowledge-base safe caches and duplicate
    # extra copies (an original is always kept). Review and AI-tagged items
    # start unchecked so the user opts in.
    return (it.tier == "duplicate") or (it.tier == "safe" and it.source != "ai")


def _short_path(path: str) -> str:
    return "~" + path[len(HOME):] if path.startswith(HOME) else path


_TIER_HEADINGS = {
    "safe": "SAFE — regenerates automatically",
    "duplicate": "DUPLICATES — one original is always kept",
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
    choices = []
    last_tier = None
    for it in ordered:
        if it.tier != last_tier:
            if last_tier is not None:
                choices.append(Separator(" "))
            choices.append(Separator(f"── {_TIER_HEADINGS.get(it.tier, it.tier.upper())} ──"))
            last_tier = it.tier
        choices.append(Choice(title=_choice_title(it, path_budget), value=it,
                              checked=_default_checked(it)))

    selected = questionary.checkbox(
        "Select what to move to Trash:",
        choices=choices,
        pointer="❯",
        instruction="(↑↓ move · space = tick · a = all · i = invert · enter = confirm)",
        style=style,
    ).ask()
    return selected or []


def _select_prompts(actionable):
    """Fallback selection: bulk prompts + per-item y/N (used when no TTY)."""
    queue = []
    safe_bulk = [i for i in actionable if i.tier == "safe" and i.source != "ai"]
    dups = [i for i in actionable if i.tier == "duplicate"]
    rest = [i for i in actionable if i not in safe_bulk and i not in dups]

    if safe_bulk:
        total = sum(i.size for i in safe_bulk)
        if _ask(f"\nMove all {len(safe_bulk)} known-safe items ({human(total)}) to Trash? [y/N] ") == "y":
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
    total = sum(i.size for i in queue)
    print(f"\n{'─' * 60}")
    print(f"About to move {len(queue)} item(s) — {human(total)} — to the Trash:\n")
    risky = 0
    for it in queue:
        _, tag = TIER_STYLE[it.tier]
        line = f"  {tag} {human(it.size):>9}  {it.path}"
        risk = git_risk(it.path) if it.kind != "duplicate" else None
        if risk:
            line += f"\n      ⚠️  {risk}"
            risky += 1
        print(line)
    if risky:
        print(f"\n  ⚠️  {risky} item(s) above carry a git warning — review them.")
    print("\nEverything goes to the Trash and can be recovered from there.")

    confirm = _ask(f"\nType 'yes' to move these {len(queue)} item(s) to the Trash: ")
    if confirm != "yes":
        print("Aborted. Nothing was deleted.")
        return 0

    freed = 0
    for it in queue:
        ok, reason = is_safe_to_trash(it.path)   # safety floor, re-checked at execution
        if not ok:
            print(f"  skipped (protected): {it.path} — {reason}")
            continue
        try:
            trash(it.path)
            freed += it.size
            print(f"  trashed: {it.path}")
        except Exception as e:
            print(f"  FAILED: {it.path} — {e}")

    print(f"\nDone. Moved ~{human(freed)} to the Trash (recoverable from there).")
    print(f"Disk: {_free_space()}")
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
            "  disksage clean        pick items from a checklist to move to Trash\n"
            "  disksage setup        choose an AI provider / paste a key\n"
            "  disksage clean --min-size 5 --no-ai\n"
        ))
    p.add_argument("--version", action="version", version=f"disksage {__version__}")
    sub = p.add_subparsers(dest="cmd", metavar="{scan,clean,setup}")

    sub.add_parser("setup", help="choose an AI provider and save your key")

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
        sp.add_argument("--no-ai", action="store_true")
        sp.add_argument("--root", default=None)

    args = p.parse_args(argv)
    config.load()

    if args.cmd == "setup":
        return cmd_setup(args)

    if getattr(args, "provider", None):
        os.environ["DISKSAGE_PROVIDER"] = args.provider

    # No subcommand → behave like `scan`.
    if not hasattr(args, "min_size"):
        args = p.parse_args(["scan"])

    _maybe_first_run_setup(args)

    if args.cmd == "clean":
        return cmd_clean(args)
    return cmd_scan(args)


if __name__ == "__main__":
    sys.exit(main())
