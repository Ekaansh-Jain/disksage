# disksage

**Get back the disk space your Mac's Storage screen hides, without breaking anything.**

disksage finds what fills a developer's Mac: the same model downloaded by Ollama, LM Studio and Hugging Face, virtualenvs and `node_modules` from projects you haven't opened in months, and Xcode and Docker leftovers. It explains every item, deletes only caches that rebuild themselves, and sends your files to the Trash.

<!-- Record demo/demo.gif with demo/demo.tape (it uses a fake home folder) and put it here:
<p align="center"><img src="demo/demo.gif" alt="disksage finding the same model stored by three apps" width="760"></p>
-->

## Install

Needs macOS and Python 3.10 or newer.

```bash
pipx install disksage
```

Or run it once without installing anything: `uvx disksage`. To install the latest code from GitHub: `pipx install git+https://github.com/Ekaansh-Jain/disksage.git`.

## Use

```bash
disksage            # read-only report, deletes nothing
disksage clean      # tick what to remove: caches are deleted, your files go to the Trash
disksage models     # every local model, where it lives, and which ones are stored twice
disksage dedupe     # store duplicate models once, as APFS clones; every app keeps its file
```

Add `--dry-run` to see what `clean` or `dedupe` would do without changing anything, and `--json` to `scan` or `models` for scripts and agents.

## What's different about it

- **Duplicate models across apps.** Ollama, LM Studio and Hugging Face each keep their own copy of the same weights. disksage matches them (Ollama and Hugging Face name files by SHA-256, so most matches need no hashing) and `dedupe` turns the extra copies into APFS clones. Every app keeps a working file at the same path, and the bytes are stored once.
- **Numbers you can trust.** Sizes are bytes on disk. Hard links are counted once, sparse files count only what they use, and iCloud placeholders are skipped (they free nothing, and reading them would download them). After `clean` or `dedupe`, the last line is the measured change in free space.
- **It explains what it finds.** A built-in list covers known caches. Anything else can go to an optional AI (a cloud provider, or a local model through Ollama or LM Studio) that sees only paths and sizes. The AI can explain an item but can never mark it safe to delete.
- **A safety floor the AI can't override.** Only your home folder. Never keychains, app containers, Mail, Messages, Photos and other libraries, iCloud Drive, cloud-sync folders, iPhone backups, credentials, or anything inside a macOS package. Folders with git-tracked files or unpushed commits are flagged and never pre-ticked.
- **Your tools' own cleanup commands.** `disksage scan` shows Docker's own space report (its disk image is invisible to a file scan) and lists commands like `brew cleanup`, `uv cache prune`, `hf cache prune` and `xcrun simctl delete unavailable` that free space while keeping what you still use. `disksage clean` offers to run `docker system prune` after asking.

## How it decides

```text
scan  ->  built-in list  ->  optional AI labels for the rest  ->  safety floor  ->  you choose
```

| Tier | Meaning | In `clean` |
|---|---|---|
| SAFE | Rebuilds itself (built-in list only) | Ticked; deleted directly |
| DUPLICATE | Identical to another file | Unticked; goes to the Trash |
| REVIEW | Reclaimable, but you'd re-download or rebuild it | Unticked; goes to the Trash |
| KEEP | Your data | Never offered |

Project folders such as `node_modules`, `.venv` and `build` are only ticked when the project hasn't been touched for 30 days, and they go to the Trash. Model caches are REVIEW: re-downloading them costs gigabytes, so use `disksage dedupe` or each app's own remove command instead.

## AI labels (optional)

Run `disksage setup` and pick a provider: Groq or Google Gemini (free cloud APIs, paste a key once), a local model server (Ollama, LM Studio, llama.cpp, Jan, … are detected automatically), or none. Without an AI, unrecognized folders simply stay REVIEW. `disksage doctor` shows which provider is active.

## Privacy

- Scanning happens on your Mac. Nothing is uploaded unless you set up a cloud AI provider.
- With a cloud provider, only folder paths (your home folder shown as `~`) and sizes are sent, to the provider you picked. With a local model, nothing leaves your Mac.
- No telemetry.

## When to use something else

- For an all-in-one Mac cleaner with an app uninstaller and system maintenance, use [Mole](https://github.com/tw93/Mole).
- To clear only `node_modules`, `npx npkill` is quick.
- To turn duplicates of all your files (photos, documents) into clones, look at [Hyperspace](https://hypercritical.co/hyperspace/).

disksage focuses on developer and local-AI clutter, and on explaining what it finds.

## Contributing

The most useful contribution is teaching disksage a folder it doesn't recognize, which takes about five minutes: see [CONTRIBUTING.md](CONTRIBUTING.md). If it ever offers to delete something it shouldn't, please report it privately ([SECURITY.md](SECURITY.md)).

## License

MIT
