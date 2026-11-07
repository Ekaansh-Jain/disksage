"""disksage — an AI-aware, safety-first disk cleaner for macOS."""

from importlib.metadata import PackageNotFoundError, version

try:
    # Single source of truth: the version in pyproject.toml.
    __version__ = version("disksage")
except PackageNotFoundError:  # running from a source checkout that isn't installed
    __version__ = "0.0.0+unknown"
