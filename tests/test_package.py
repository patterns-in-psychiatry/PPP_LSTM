"""A single smoke check for the installed core package."""

import subprocess
import sys


def test_core_import_without_tensorflow(tmp_path):
    result = subprocess.run(
        [
            sys.executable,
            "-I",
            "-c",
            "import ppp; import numpy; import pandas; import sklearn; import sys; "
            "assert 'tensorflow' not in sys.modules; "
            "assert 'keras' not in sys.modules",
        ],
        cwd=tmp_path,
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 0, result.stderr
