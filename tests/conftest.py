"""Point HOME at a throwaway folder before any disksage module is imported.

disksage builds its paths (scan roots, protected folders) from HOME when its
modules load, so this keeps the tests from reading or writing your real home
folder, and lets them create fake ~/Library, ~/.cache, ... trees freely.
"""

import atexit
import os
import shutil
import tempfile

import pytest

_fake_home = tempfile.mkdtemp(prefix="disksage-test-home-")
os.environ["HOME"] = _fake_home
atexit.register(shutil.rmtree, _fake_home, ignore_errors=True)


@pytest.fixture(autouse=True)
def _fresh_scan_and_no_docker(monkeypatch):
    """Every test starts without a remembered scan (so `clean` scans what the
    test just created), and without Docker: CI machines may run a real Docker,
    which would add a Docker entry to every checklist."""
    from disksage import models, tools

    path = os.path.join(models.STATE_DIR, "last-scan.json")
    if os.path.exists(path):
        os.remove(path)
    monkeypatch.setattr(tools, "docker_report", lambda timeout=5: None)
