import os
import subprocess
import sys

import pytest

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))

SCRIPTS = [
    'run_demo_RGB_Depth.py',
    'run_demo_RGB_Disparity.py',
    'trainInconSeg_with_RGB_Depth.py',
    'trainInconSeg_with_RGB_Disparity.py',
]


@pytest.mark.parametrize('script', SCRIPTS)
def test_script_help(script):
    proc = subprocess.run(
        [sys.executable, os.path.join(ROOT, script), '--help'],
        cwd=ROOT,
        capture_output=True,
        text=True,
    )
    assert proc.returncode == 0, proc.stderr
