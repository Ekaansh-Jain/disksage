"""User configuration — where a person's AI choice + keys are stored.

Kept in ~/.config/disksage/.env (chmod 600) so it persists across runs and
isn't tied to the current directory. Variables already set in the shell win
over the file. A .env in the current folder is never read: any repo you
clone can contain one, and it must not decide where your folder names go.

Keys the user pastes during `disksage setup` are written here and never printed
back. Nothing is ever sent anywhere except the AI provider they chose.
"""

from __future__ import annotations

import contextlib
import os

CONFIG_DIR = os.path.expanduser("~/.config/disksage")
ENV_PATH = os.path.join(CONFIG_DIR, ".env")


def load() -> None:
    """Load the user config. Variables already set in the shell win."""
    try:
        from dotenv import load_dotenv
    except ImportError:
        return
    if os.path.exists(ENV_PATH):
        load_dotenv(ENV_PATH)


def is_configured() -> bool:
    """True once the user has completed setup at least once."""
    return os.path.exists(ENV_PATH)


def save(values: dict[str, str]) -> str:
    """Write config values (creating the dir), apply them to this process, and
    return the file path. Overwrites any previous config."""
    os.makedirs(CONFIG_DIR, mode=0o700, exist_ok=True)
    body = "# disksage configuration — created by `disksage setup`\n"
    body += "# You can edit this file by hand; keep it private.\n"
    for k, v in values.items():
        body += f"{k}={v}\n"
    # The key goes into a new file that only you can read, which then replaces
    # the old one. Writing first and running chmod afterwards would leave a
    # moment where other accounts on the Mac could read it.
    target = os.path.realpath(ENV_PATH)  # a symlinked config (dotfiles) stays a symlink
    tmp = f"{target}.{os.getpid()}.tmp"
    try:
        with open(os.open(tmp, os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o600), "w") as f:
            f.write(body)
        os.replace(tmp, target)
    except BaseException:
        with contextlib.suppress(OSError):
            os.remove(tmp)
        raise
    if "DISKSAGE_AI" not in values:
        os.environ.pop("DISKSAGE_AI", None)  # a provider was chosen: the AI is on again
    for k, v in values.items():
        os.environ[k] = v
    return ENV_PATH
