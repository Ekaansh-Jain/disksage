# Contributing to disksage

Thanks for helping. The most useful contribution is teaching disksage about a folder it doesn't recognize, and it takes about five minutes.

## Set up

```bash
git clone https://github.com/Ekaansh-Jain/disksage.git
cd disksage
python3 -m venv .venv && source .venv/bin/activate
pip install -e ".[dev]"
python -m pytest -q
ruff check .
```

`disksage scan` is read-only, so it's safe to run on your own Mac while you work. To try `clean` without risk, build the fake home folder from `demo/make_demo_home.sh` and run `HOME=/tmp/disksage-demo disksage clean`.

## Add a rule

Rules live in `disksage/knowledge.py`. Add a `Rule(...)` above the broad fallbacks at the bottom of the list:

```python
Rule("safe", "Bun cache", "Bun re-downloads packages on the next install",
     path_glob=_h(".bun/install/cache")),
```

Pick the tier carefully:

| Tier | Use it when | Examples |
|---|---|---|
| SAFE | It rebuilds itself and nothing is lost | pip cache, Xcode DerivedData |
| REVIEW | It can come back, but costs a re-download, a rebuild or a login | `node_modules`, Ollama models |
| KEEP | It's real data | Mail, Messages, app databases |

If you're not sure, use REVIEW. Add a test case for the rule in `tests/`, then open a pull request and say how you know (a link to the tool's docs is best).

## Safety rules

disksage deletes files, so a few rules don't bend:

- Don't weaken anything in `disksage/safety.py`. If a protection blocks something legitimate, open an issue first.
- SAFE means "rebuilds itself". Re-downloadable isn't enough.
- The AI may label and explain folders. It never marks anything SAFE.
- Every change that touches deletion needs a test.

## Pull requests

Keep pull requests small and focused, and fill in the checklist in the template. For bigger changes, open an issue first so we can agree on the approach before you spend time on it.

## Safety problems

If disksage offered to delete something it shouldn't have, please report it privately. See [SECURITY.md](SECURITY.md).
