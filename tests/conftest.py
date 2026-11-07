"""Point HOME at a throwaway folder before any disksage module is imported.

disksage builds its paths (scan roots, protected folders) from HOME when its
modules load, so this keeps the tests from reading or writing your real home
folder, and lets them create fake ~/Library, ~/.cache, ... trees freely.
"""

import atexit
import os
import shutil
import tempfile

_fake_home = tempfile.mkdtemp(prefix="disksage-test-home-")
os.environ["HOME"] = _fake_home
atexit.register(shutil.rmtree, _fake_home, ignore_errors=True)
