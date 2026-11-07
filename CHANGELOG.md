# Changelog

Notable changes to disksage. The format follows [Keep a Changelog](https://keepachangelog.com/en/1.1.0/), and versions follow [Semantic Versioning](https://semver.org/).

## [Unreleased]

<!-- When you tag v0.2.0, rename this heading to "## [0.2.0] - <date>". -->

### Added

- `disksage` in a terminal scans, shows a summary, and asks "Review and clean now?", then goes straight to the checklist using the same scan. Piped, it stays the read-only report.
- The summary groups what can be freed by what it costs you (nothing lost, nothing deleted, rebuild if needed, re-download if needed, Docker's own cleanup, yours to judge), then lists the 8 biggest items. `scan --details` shows every item.
- A live file and byte counter while scanning, "Scanned N files in X s" at the end, a spinner per item while cleaning, and progress bars while hashing model files.
- `clean` reuses a scan from the last 10 minutes (`--rescan` to scan again). Only paths and sizes are reused; tiers are always worked out again from the rules.
- Docker's cleanup is a checklist item, with Docker's own numbers.
- `models --paths` and `dedupe --paths` show where files are; by default you see model and file names instead of hashes.
- `disksage models` lists every model file across Ollama, LM Studio and the Hugging Face cache, and finds the same file stored by more than one app.
- `disksage dedupe` replaces duplicate model files with APFS clones: every app keeps a working file, and the bytes are stored once. It verifies contents by SHA-256 first, swaps files atomically, and remembers the clones it made.
- `--dry-run` for `clean` and `dedupe`, and `--json` for `scan` and `models`.
- Docker's own space report in `scan`, and `clean` offers `docker system prune` after asking. `scan` also lists other tools' cleanup commands (Homebrew, Xcode simulators, uv, pnpm, conda, Hugging Face).
- Folders marked with a `CACHEDIR.TAG` file are recognized as caches.
- ML leftovers in projects (`wandb`, `mlruns`, `lightning_logs`, `.ipynb_checkpoints`) are listed as REVIEW.
- CI on macOS and Ubuntu, a release workflow that publishes to PyPI, issue forms, a PR template, CONTRIBUTING.md and SECURITY.md.

### Fixed

- Sizes are bytes on disk, so sparse files and hard links no longer inflate totals.
- A hard-linked file is no longer offered as a duplicate of itself.
- iCloud and other cloud placeholders are skipped, and never downloaded during duplicate checks.
- Hugging Face and PyTorch caches are REVIEW instead of pre-ticked SAFE, and the Hugging Face login token is never offered.
- `clean` frees space right away: caches that rebuild themselves are deleted, your files still go to the Trash, and on macOS you can empty the Trash at the end. The last line shows the measured change in free space.
- `~/.Trash` is no longer offered as an item (it can't be moved into itself).
- Nothing inside a macOS package (Photos libraries, apps, Final Cut and Logic projects, VM bundles) is scanned or offered.
- Folders with git-tracked files get a warning and are never pre-ticked.
- `.turbo` is labelled "Turbo cache".

### Changed

- Paths are shown as `~/…`. Long ones lose their middle but keep whole folder names and the project name.
- Project folders are walked once instead of twice, and Docker and the duplicate-model estimate are checked while the scan runs instead of after it.
- Ctrl+C stops a scan cleanly, and stops `clean` after the current item with a count of what was done.
- The AI labels and explains folders but can never mark anything SAFE. Paths are sent as `~/…`, so your username isn't shared.
- Duplicates start unticked, and project folders like `node_modules` are pre-ticked only when the project hasn't been touched for 30 days.
- The first run shows results straight away; the scan suggests `disksage setup` when an AI could identify more.
- Protected: iCloud Drive, cloud-sync folders, iPhone backups and credential folders (`~/.ssh`, `~/.aws`, `~/.config`, …).
- Dropped the `openai` dependency: one standard-library HTTP request talks to every provider, and every dependency is pure Python.
- Requires Python 3.10 or newer.

## [0.1.0] - 2026-09-29

- First release.
