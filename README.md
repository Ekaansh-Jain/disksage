# disksage

An AI-aware, safety-first disk cleaner for macOS. It finds where your space
actually went, **explains what each thing is in plain language**, and only ever
moves approved items to the Trash — never `rm -rf`, never without asking.

Most disk tools just show you a big anonymous blob and let you delete blindly.
`disksage` recognises the long tail — the Arduino cores, the browser caches, the
bloated `.git`, the `node_modules` graveyard — tells you *why* each one is safe
or risky, and refuses to touch anything it shouldn't.

## How it works

```
scan  →  AI identifies & explains  →  safety floor  →  you review  →  Trash
         (handles the weird stuff)    (rules override    (safe/review   (reversible)
                                        the AI)            /keep)
```

1. **Scan** — deterministic `du`/`find`-style scan of the places space hides
   (caches, app support, project `node_modules`/build dirs, known heavy folders).
2. **Knowledge base first** — a baked-in list of known paths classifies most
   items instantly, offline and free (`disksage/knowledge.py`).
3. **AI for the unknowns** — anything unrecognised is sent to an LLM, **paths
   and sizes only, never file contents**, to be labelled and explained.
4. **Safety floor** — a dumb, unbreakable layer under the AI: only inside your
   home dir, never protected paths, warns on git repos with unpushed work, and
   deletes only to the Trash (`disksage/safety.py`).

The AI proposes; the safety floor disposes. The tool runs fully **without any
AI** — it just falls back to the knowledge base.

## Requirements

- **macOS** (the path knowledge targets macOS layouts)
- **Python 3.9+** (`python3 --version`)
- Optionally an AI key (Groq/Gemini) or LM Studio — the tool works without one too

## Install

The recommended way (installs the `disksage` command in its own isolated env):

```bash
pipx install git+https://github.com/Ekaansh-Jain/disksage.git
```

Don't have `pipx`? Install it once with `brew install pipx && pipx ensurepath`
(or use plain `pip` below). Then just run `disksage`.

Or from a local clone:

```bash
cd disksage
pipx install .          # or: python3 -m pip install .
```

Or run without installing:

```bash
cd disksage
python3 -m pip install -r requirements.txt
python3 -m disksage
```

## Use

```bash
disksage               # scan + report, deletes nothing
disksage scan          # same
disksage clean         # scan, then pick items from a checklist to move to Trash
```

`disksage clean` opens an interactive **checklist** — arrow keys to move, space to
tick, enter to confirm — with safe caches and duplicate copies pre-ticked and
everything else left for you to opt in. It then shows the full list and asks you
to type `yes` before anything moves. (When input is piped or non-interactive, it
falls back to per-item yes/no prompts.)

Options:
- `--min-size 5` — folder size threshold in MB (default 20)
- `--big-size 500` — loose-file threshold in MB (default 200)
- `--dup-size 100` — duplicate-detection threshold in MB (default 50)
- `--no-files` — skip big-file + duplicate scanning
- `--no-ai` — knowledge base only
- `--provider groq|gemini|local`

## What it finds

- **Caches** — known regenerable caches (pip, npm, browsers, Xcode, …) → 🟢 safe
- **Buried caches** — cache subfolders hidden inside app data (Chrome, Electron, Claude sandbox) → surfaced separately
- **Project cruft** — `node_modules`, `.venv`, `build`, `.next`, `target`, … → review
- **Model stores** — `~/.lmstudio`, `~/.ollama`, HuggingFace/torch caches
- **Big loose files** — models/datasets/media/archives sitting in your projects → review (your data)
- **Duplicates** — byte-identical copies, found by hashing; one original is always kept

## Testing

```bash
python3 -m pip install -e ".[dev]"
python3 -m pytest -q
```

The suite proves the safety guarantees (protected-path blocking, exact
duplicate grouping, non-overlapping sizing) so they hold on any machine.

## First-run setup

The first time you run `disksage`, it asks how you want unknown files labelled:

```
👋 Welcome to disksage — quick one-time setup.

Choose an AI provider:
❯ Groq — free cloud API (paste a key)
  Google Gemini — free cloud API (paste a key)
  Local LM Studio — private, no key (needs the app running)
  No AI — use the built-in knowledge base only
```

Pick one; if it needs a key you paste it once (hidden input) and it's saved to
`~/.config/disksage/.env` (chmod 600). Re-run anytime with `disksage setup`, or
set the keys yourself — auto-detect order is **local → Groq → Gemini**:

- **LM Studio** (local, private, offline) — start its server; auto-detected on `:1234`.
- **Groq** — `GROQ_API_KEY`.
- **Gemini** — `GEMINI_API_KEY`.

Keys stay on your machine; only paths + sizes are ever sent to the provider you
chose. The tool works fully with **No AI** too — you just lose labels on
unrecognized folders.

## Safety

Designed so it's safe for anyone to run on their own machine:

- **Scan is read-only.** `disksage` (no subcommand) never deletes anything.
- **Deletes go to the Trash** via `send2trash` — never `rm -rf`. Recover
  anything.
- **Protected paths can never be removed** — home itself, top-level folders
  (Documents, Desktop, …), keychains, app containers, and photo libraries are
  blocked deterministically, even if the AI mislabels them.
- **Only inside your home directory** — nothing outside `~` is ever touched.
- **The AI never has delete authority.** Only *knowledge-base* items are
  offered for bulk deletion; anything the AI tagged safe requires per-item
  confirmation.
- **Bulk deletion is limited** to known-safe caches and duplicate *extra*
  copies. Everything else is confirmed one item at a time.
- **Duplicates keep an original.** Only extra copies are ever offered; the tool
  can't remove your last copy of a file.
- **Git-aware.** A repo with unpushed commits or uncommitted changes triggers a
  warning before removal.
- **Privacy.** Only paths + sizes are ever sent to an LLM — never file contents.

## Platform

macOS-focused (the knowledge base and protected paths target macOS layouts).
It won't harm other systems — it just finds less — but Linux/Windows support
means extending `knowledge.py` and `safety.py`.

## Contributing

The heart of the project is the **knowledge base** (`disksage/knowledge.py`) —
the list of "what is this folder and is it safe to delete." The easiest and most
valuable contribution is teaching it a new path:

1. Fork and clone, then `python3 -m pip install -e ".[dev]"`.
2. Add a `Rule(...)` in `knowledge.py` for the cache/app you know about.
3. Run the tests: `python3 -m pytest -q` (they enforce the safety guarantees).
4. Open a pull request.

Please keep the safety rules intact — anything uncertain should be tagged
`review`, never `safe`, and protected paths in `safety.py` must stay protected.

## License

MIT.
