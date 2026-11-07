## What changed


## How I tested it

- [ ] `python -m pytest -q` passes
- [ ] `ruff check .` passes
- [ ] Ran `disksage scan` on a real Mac (macOS version: )

## Safety checklist (rule or deletion changes)

- [ ] New SAFE rules only cover data that rebuilds itself, and the "why" says how
- [ ] Anything I'm unsure about is REVIEW, not SAFE
- [ ] No protection in `safety.py` got weaker
- [ ] Every new rule has a test case
